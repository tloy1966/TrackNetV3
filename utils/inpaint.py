import math

import numpy as np
import torch
from utils.runtime import StageTimer


def get_ensemble_weight(seq_len, eval_mode):
    """Return uniform or positional weights for a temporal ensemble."""
    if seq_len < 1:
        raise ValueError('Sequence length must be positive')
    if eval_mode == 'average':
        return torch.ones(seq_len) / seq_len
    if eval_mode != 'weight':
        raise ValueError(f'Invalid ensemble mode: {eval_mode}')
    weight = torch.ones(seq_len)
    for i in range(math.ceil(seq_len / 2)):
        weight[i] = i + 1
        weight[seq_len - i - 1] = i + 1
    return weight / weight.sum()


def generate_inpaint_mask(pred_dict, th_h=30, seq_len=None):
    """Select bounded missing runs, optionally limited to a model window."""
    y = np.asarray(pred_dict['Y'])
    visibility = np.asarray(pred_dict['Visibility'])
    if y.ndim != 1 or visibility.shape != y.shape:
        raise ValueError('Coordinates and visibility must have equal lengths')
    if not np.isfinite(y).all() or not np.isin(visibility, [0, 1]).all():
        raise ValueError('Coordinates must be finite and visibility must contain only 0 or 1')
    frames = np.asarray(pred_dict['Frame']) if 'Frame' in pred_dict else np.arange(len(y))
    if frames.shape != y.shape:
        raise ValueError('Frame indices and coordinates must have equal lengths')
    if seq_len is not None and seq_len < 1:
        raise ValueError('Sequence length must be positive')

    mask = np.zeros(len(y), dtype=np.int64)
    start = 0
    while start < len(y):
        if visibility[start] != 0:
            start += 1
            continue
        end = start
        while end < len(y) and visibility[end] == 0:
            end += 1
        if (start > 0 and end < len(y)
                and visibility[start - 1] == visibility[end] == 1
                and y[start - 1] > th_h and y[end] > th_h
                and np.all(np.diff(frames[start - 1:end + 1]) == 1)
                and (seq_len is None or end - start <= seq_len - 2)):
            mask[start:end] = 1
        start = end
    return mask.tolist()


def inpaint_batch(model, coordinates, inpaint_mask, indices=None, coor_th=0., timer=None):
    """Repair only zero coordinates bracketed by observations in each window.

    Coordinates have shape (N, L, 2), masks (N, L, 1), and optional indices
    (N, L, 2) contain rally and frame IDs. Returned masks indicate eligible
    repairs, not confidence or successful detections.
    """
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError('Coordinates must have shape (N, L, 2)')
    if not coordinates.is_floating_point():
        raise ValueError('Coordinates must be floating-point normalized values')
    if inpaint_mask.shape != coordinates.shape[:2] + (1,):
        raise ValueError('Inpaint mask must have shape (N, L, 1)')
    if not torch.isfinite(coordinates).all():
        raise ValueError('Input coordinates must be finite')
    if not ((inpaint_mask == 0) | (inpaint_mask == 1)).all():
        raise ValueError('Inpaint mask must contain only 0 or 1')

    batch_size, seq_len = coordinates.shape[:2]
    if seq_len < 1:
        raise ValueError('Sequence length must be positive')
    positions = torch.arange(seq_len, device=coordinates.device).expand(batch_size, -1)
    observed = (coordinates != 0).any(dim=-1)
    left = torch.where(observed, positions, -1).cummax(dim=1).values
    right = torch.where(observed, positions, seq_len).flip(dims=(1,)).cummin(dim=1).values.flip(dims=(1,))
    requested = inpaint_mask[..., 0].bool() & ~observed
    eligible = requested & (left >= 0) & (right < seq_len)

    if indices is not None:
        indices = torch.as_tensor(indices, device=coordinates.device)
        if indices.shape != coordinates.shape[:2] + (2,):
            raise ValueError('Indices must have shape (N, L, 2)')
        consecutive = ((indices[:, 1:, 0] == indices[:, :-1, 0])
                       & (indices[:, 1:, 1] == indices[:, :-1, 1] + 1))
        breaks = torch.cat((torch.zeros((batch_size, 1), device=coordinates.device, dtype=torch.long),
                            (~consecutive).long().cumsum(dim=1)), dim=1)
        eligible &= (breaks.gather(1, left.clamp(min=0))
                     == breaks.gather(1, right.clamp(max=seq_len - 1)))

    result = coordinates.clone()
    active = eligible.any(dim=1)
    if active.any():
        active_mask = eligible[active, :, None]
        model_coordinates = coordinates[active]
        model_mask = requested[active, :, None].to(coordinates.dtype)
        timer = timer if timer is not None else StageTimer()
        with timer.measure('inpaintnet_model', coordinates.device):
            prediction = model(model_coordinates, model_mask)
        timer.count('inpaintnet_windows', len(model_coordinates))
        timer.count('inpaintnet_model_frame_evaluations', len(model_coordinates) * seq_len)
        if not torch.is_tensor(prediction) or prediction.shape != coordinates[active].shape:
            raise ValueError('InpaintNet must return coordinates with the input shape')
        if not torch.isfinite(prediction).all():
            raise ValueError('InpaintNet produced non-finite coordinates')
        if ((prediction < 0) | (prediction > 1)).any():
            raise ValueError('InpaintNet coordinates must be normalized to [0, 1]')
        below_threshold = (prediction[..., 0] < coor_th) & (prediction[..., 1] < coor_th)
        prediction = prediction.masked_fill(below_threshold[..., None], 0.)
        result[active] = torch.where(active_mask, prediction, coordinates[active])
    return result, eligible[..., None]


def inpaint_trajectory(model, coordinates, inpaint_mask, seq_len, eval_mode='weight',
                       batch_size=16, device='cpu', indices=None, coor_th=0., timer=None):
    """Ensemble only windows containing both real endpoints of a missing run.

    Inputs are normalized coordinates (T, 2), a candidate mask (T, 1), and
    optional rally/frame IDs (T, 2). Short trajectories repeat their last frame
    as input padding only; no padded frame is emitted or counted as evidence.
    """
    if seq_len < 1 or batch_size < 1:
        raise ValueError('Sequence length and batch size must be positive')
    if eval_mode not in ('nonoverlap', 'average', 'weight'):
        raise ValueError(f'Invalid evaluation mode: {eval_mode}')
    coordinates = torch.as_tensor(coordinates).detach().cpu()
    inpaint_mask = torch.as_tensor(inpaint_mask).detach().cpu()
    timer = timer if timer is not None else StageTimer()
    if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
        raise ValueError('Coordinates must have shape (T, 2)')
    num_frames = len(coordinates)
    if inpaint_mask.shape != (num_frames, 1):
        raise ValueError('Inpaint mask must have shape (T, 1)')
    if indices is None:
        indices = torch.stack((torch.zeros(num_frames, dtype=torch.long),
                               torch.arange(num_frames)), dim=1)
    else:
        indices = torch.as_tensor(indices).detach().cpu()
        if indices.shape != (num_frames, 2):
            raise ValueError('Indices must have shape (T, 2)')

    result = coordinates.clone()
    weight_sum = torch.zeros(num_frames, dtype=torch.float64)
    coordinate_sum = torch.zeros((num_frames, 2), dtype=torch.float64)
    if num_frames == 0:
        return result, torch.zeros((0, 1), dtype=torch.bool)

    if eval_mode == 'nonoverlap':
        starts = list(range(0, num_frames, seq_len))
        position_weight = torch.ones(seq_len)
    else:
        starts = list(range(max(1, num_frames - seq_len + 1)))
        position_weight = get_ensemble_weight(seq_len, eval_mode)
    offsets = torch.arange(seq_len)

    for start in range(0, len(starts), batch_size):
        positions = torch.tensor(starts[start:start + batch_size])[:, None] + offsets
        real_frames = positions < num_frames
        positions = positions.clamp(max=num_frames - 1)
        with torch.no_grad():
            with timer.measure('transfer', device):
                batch_coordinates = coordinates[positions].to(device)
                batch_mask = inpaint_mask[positions].to(device)
                batch_indices = indices[positions].to(device)
            prediction, eligible = inpaint_batch(
                model, batch_coordinates, batch_mask,
                indices=batch_indices, coor_th=coor_th, timer=timer)
            with timer.measure('transfer', device):
                prediction = prediction.cpu()
                eligible = eligible[..., 0].cpu() & real_frames
        weights = position_weight.expand(len(positions), -1).double().clone()
        # Match the existing uniform ensemble at the trajectory boundaries.
        boundary = (positions < seq_len - 1) | (positions > num_frames - seq_len)
        weights[boundary] = 1.
        weights *= eligible
        coordinate_sum.index_add_(0, positions.reshape(-1),
                                  (prediction.double() * weights[..., None]).reshape(-1, 2))
        weight_sum.index_add_(0, positions.reshape(-1), weights.reshape(-1))

    supported = weight_sum > 0
    result[supported] = (coordinate_sum[supported] / weight_sum[supported, None]).to(result.dtype)
    below_threshold = (result[:, 0] < coor_th) & (result[:, 1] < coor_th)
    result[supported & below_threshold] = 0.
    return result, supported[:, None]


def format_inpaint_predictions(raw_pred, coordinates, inpaint_mask, img_shape):
    """Preserve raw pixel detections exactly and add explicit repair provenance."""
    coordinates = coordinates.detach().cpu().numpy() if torch.is_tensor(coordinates) else np.asarray(coordinates)
    inpaint_mask = inpaint_mask.detach().cpu().numpy() if torch.is_tensor(inpaint_mask) else np.asarray(inpaint_mask)
    num_frames = len(raw_pred['Frame'])
    if coordinates.shape != (num_frames, 2) or inpaint_mask.shape != (num_frames, 1):
        raise ValueError('Prediction coordinates and masks must match the raw frame count')
    if any(len(raw_pred[key]) != num_frames for key in ('X', 'Y', 'Visibility')):
        raise ValueError('Raw prediction columns must have equal lengths')
    if not np.isfinite(coordinates).all():
        raise ValueError('Prediction coordinates must be finite')
    if not np.isin(inpaint_mask, [0, 1]).all():
        raise ValueError('Inpaint mask must contain only 0 or 1')

    result = {key: list(raw_pred[key]) for key in ('Frame', 'X', 'Y', 'Visibility')}
    for key in ('X', 'Y', 'Visibility'):
        result[f'{key}_Raw'] = list(raw_pred[key])
    repair = inpaint_mask[:, 0].astype(bool) & (np.asarray(raw_pred['Visibility']) == 0)
    result['Inpaint_Mask'] = repair.astype(int).tolist()
    for i in range(num_frames):
        if raw_pred['Visibility'][i] == 0:
            x, y = (coordinates[i] * img_shape).astype(np.int64) if repair[i] else (0, 0)
            result['X'][i], result['Y'][i] = int(x), int(y)
            result['Visibility'][i] = int(x != 0 or y != 0)
    return result
