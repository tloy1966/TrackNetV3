import torch

from utils.inpaint import get_ensemble_weight
from utils.runtime import StageTimer


def iter_heatmap_predictions(model, data_loader, eval_mode, device, timer=None):
    """Reuse temporal weights while emitting each real frame exactly once."""
    if eval_mode not in ('nonoverlap', 'average', 'weight'):
        raise ValueError(f'Invalid evaluation mode: {eval_mode}')
    timer = timer if timer is not None else StageTimer()
    pending = {}
    last_start = None

    def emit(key, trailing=False):
        index, true, votes, seq_len = pending.pop(key)
        if len(votes) == seq_len and eval_mode == 'weight':
            prediction = (torch.stack(votes) * get_ensemble_weight(seq_len, 'weight')[:, None, None]).sum(0)
        else:
            padding = [torch.zeros_like(votes[0])] * (seq_len - len(votes))
            values = votes + padding if trailing else padding + votes
            prediction = torch.stack(values).sum(0) / len(votes)
        return index, true, prediction

    def batch_result(items):
        indices = torch.stack([item[0] for item in items])[:, None, :]
        truth = None if items[0][1] is None else torch.stack([item[1] for item in items])[:, None, :, :]
        prediction = torch.stack([item[2] for item in items])[:, None, :, :]
        return indices, truth, prediction

    iterator = iter(data_loader)
    while True:
        with timer.measure('input'):
            try:
                batch = next(iterator)
            except StopIteration:
                break
        indices, frames = batch[:2]
        truth = batch[2] if len(batch) > 2 else None
        with timer.measure('transfer', device):
            frames = frames.float().to(device)
        with torch.no_grad(), timer.measure('tracknet_model', device):
            prediction = model(frames)
        with timer.measure('transfer', device):
            prediction = prediction.detach().cpu()
        batch_size, seq_len = indices.shape[:2]
        timer.count('tracknet_windows', batch_size)
        timer.count('tracknet_model_frame_evaluations', batch_size * seq_len)
        emitted = []
        with timer.measure('temporal_ensemble'):
            for n in range(batch_size):
                start = tuple(int(value) for value in indices[n, 0])
                if last_start is not None and start <= last_start:
                    raise ValueError('Input windows must have strictly increasing start indices')
                last_start = start
                previous = None
                for f in range(seq_len):
                    key = tuple(int(value) for value in indices[n, f])
                    if key == previous:
                        break
                    if previous is not None and key <= previous:
                        raise ValueError('Input frames must be increasing, with padding only at the end')
                    previous = key
                    true = truth[n, f] if truth is not None else None
                    if eval_mode == 'nonoverlap':
                        emitted.append((indices[n, f], true, prediction[n, f]))
                    else:
                        if key not in pending:
                            pending[key] = (indices[n, f], true, [], seq_len)
                        pending[key][2].append(prediction[n, f].clone())
                timer.maximum('peak_pending_frames', len(pending))
                if eval_mode != 'nonoverlap':
                    for key in list(pending):
                        if key <= start:
                            emitted.append(emit(key))
        if emitted:
            yield batch_result(emitted)
    if pending:
        with timer.measure('temporal_ensemble'):
            emitted = [emit(key, trailing=True) for key in list(pending)]
        yield batch_result(emitted)
