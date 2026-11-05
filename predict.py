import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from test import predict_location, evaluate_prediction_dict
from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from utils.general import *
from utils.metric import metrics_from_types, validate_heatmap_threshold
from utils.runtime import (
    StageTimer, evaluation_protocol, file_identity, load_checkpoint,
    video_result, write_run_report,
)
from utils.temporal import iter_heatmap_predictions
from utils.inpaint import (
    format_inpaint_predictions, generate_inpaint_mask,
    get_ensemble_weight, inpaint_trajectory,
)

# -----------------------------------------------------------------------------
# PyTorch inference device
# -----------------------------------------------------------------------------
#
# TrackNetV3 originally assumes an NVIDIA GPU and uses CUDA directly.
# Select the best available PyTorch backend so inference can also run on
# Apple Silicon through MPS, while preserving CUDA and CPU support.
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using PyTorch device: {device}")


def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1), heatmap_threshold=0.5):
    """ Predict coordinates from heatmap or inpainted coordinates. 

        Args:
            indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)
            heatmap_threshold (float): finite [0,1] strict heatmap decoding threshold

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy() if torch.is_tensor(indices) else np.asarray(indices)
    
    # Transform input for heatmap prediction
    if y_pred is not None:
        y_pred = y_pred > validate_heatmap_threshold(heatmap_threshold)
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred) # (N, L, H, W)
    
    # Transform input for coordinate prediction
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    # Predict from coordinate
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT* img_scaler[1]) 
                elif y_pred is not None:
                    # Predict from heatmap
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    cx_pred, cy_pred = int(bbox_pred[0]+bbox_pred[2]/2), int(bbox_pred[1]+bbox_pred[3]/2)
                    cx_pred, cy_pred = int(cx_pred*img_scaler[0]), int(cy_pred*img_scaler[1])
                else:
                    raise ValueError('Invalid input')
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break
    
    return pred_dict    


def predict_video(tracknet, video_file, settings, inpaintnet=None, timer=None, inference_device=None):
    timer = timer if timer is not None else StageTimer()
    inference_device = device if inference_device is None else inference_device
    seq_len, bg_mode = settings['tracknet_seq_len'], settings['bg_mode']
    nonoverlap = settings['eval_mode'] == 'nonoverlap'
    with timer.measure('input_setup'):
        if settings['large_video']:
            dataset = Video_IterableDataset(
                video_file, seq_len=seq_len, sliding_step=seq_len if nonoverlap else 1,
                bg_mode=bg_mode, max_sample_num=settings['max_sample_num'],
                video_range=settings['video_range'], timer=timer)
            w, h, fps, declared_frames = dataset.w, dataset.h, dataset.fps, dataset.video_len
            data_loader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=False, drop_last=False)
        else:
            cap = cv2.VideoCapture(video_file)
            try:
                if not cap.isOpened():
                    raise ValueError(f'Cannot open video: {video_file}')
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps, declared_frames = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                cap.release()
            with timer.measure('decode'):
                frame_list = generate_frames(video_file)
            timer.count('decoded_frames', len(frame_list))
            if frame_list:
                if w <= 0 or h <= 0:
                    raise ValueError('Nonempty video has invalid image dimensions')
                step = seq_len if nonoverlap or len(frame_list) < seq_len else 1
                dataset = Shuttlecock_Trajectory_Dataset(
                    seq_len=seq_len, sliding_step=step, data_mode='heatmap', bg_mode=bg_mode,
                    frame_arr=np.asarray(frame_list)[..., ::-1], padding=step == seq_len)
                data_loader = DataLoader(
                    dataset, batch_size=settings['batch_size'], shuffle=False,
                    num_workers=settings['num_workers'], drop_last=False)
            else:
                dataset, data_loader = None, []
    img_scaler = (w / WIDTH, h / HEIGHT)
    raw = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    tracknet.eval()
    for indices, _, heatmaps in iter_heatmap_predictions(
            tracknet, tqdm(data_loader), settings['eval_mode'], inference_device, timer):
        with timer.measure('postprocess'):
            prediction = predict(indices, y_pred=heatmaps, img_scaler=img_scaler,
                                 heatmap_threshold=settings['heatmap_threshold'])
            for key, values in prediction.items():
                raw[key].extend(values)
    if inpaintnet is not None and raw['Frame']:
        inpaintnet.eval()
        with timer.measure('postprocess'):
            mask = generate_inpaint_mask(raw, th_h=h * 0.05, seq_len=settings['inpaintnet_seq_len'])
            coordinates = torch.tensor(np.column_stack((raw['X'], raw['Y'])),
                                       dtype=torch.float32) / torch.tensor((w, h))
            coordinates[torch.tensor(raw['Visibility']) == 0] = 0.
            indices = torch.stack((torch.zeros(len(coordinates), dtype=torch.long),
                                   torch.tensor(raw['Frame'], dtype=torch.long)), dim=1)
        with timer.measure('inpainting', inference_device):
            coordinates, repair_mask = inpaint_trajectory(
                inpaintnet, coordinates, torch.tensor(mask, dtype=torch.float32).reshape(-1, 1),
                settings['inpaintnet_seq_len'], eval_mode=settings['eval_mode'],
                batch_size=settings['batch_size'], device=inference_device, indices=indices,
                coor_th=COOR_TH, timer=timer)
    else:
        coordinates = torch.zeros((len(raw['Frame']), 2))
        repair_mask = torch.zeros((len(coordinates), 1), dtype=torch.bool)
    with timer.measure('postprocess'):
        final = format_inpaint_predictions(raw, coordinates, repair_mask, (w, h))
    metadata = {
        'path': video_file, 'width': w, 'height': h,
        'fps': fps if math.isfinite(fps) and fps > 0 else None,
        'declared_frames': declared_frames, 'processed_frames': len(raw['Frame']),
        'input_source': 'streamed_video' if settings['large_video'] else 'preloaded_video',
        'num_workers': data_loader.num_workers if dataset is not None else 0,
        'heatmap_threshold_applied': settings['heatmap_threshold'],
    }
    if settings['large_video']:
        metadata['streaming'] = dataset.stats
    else:
        metadata['background'] = {
            'method': 'exact_all_frame_median' if bg_mode else 'none',
            'sample_count': len(raw['Frame']) if bg_mode else 0,
            'sampling_controls_apply': False,
        }
    return raw, final, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='file path of the video')
    parser.add_argument('--tracknet_file', type=str, help='file path of the TrackNet model checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='file path of the InpaintNet model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--heatmap_threshold', type=validate_heatmap_threshold, default=0.5,
                        help='strict heatmap > threshold decoding in [0,1]; not calibrated confidence')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--max_sample_num', type=int, default=1800, help='hard maximum original-resolution background samples for --large_video')
    parser.add_argument('--video_range', type=lambda splits: [int(s) for s in splits.split(',')], default=None, help='[start,end) seconds for --large_video background sampling')
    parser.add_argument('--save_dir', type=str, default='pred_result', help='directory to save the prediction result')
    parser.add_argument('--large_video', action='store_true', default=False, help='stream cached float32 frame windows; backgrounds and outputs still require additional memory')
    parser.add_argument('--output_video', action='store_true', default=False, help='whether to output video with predicted trajectory')
    parser.add_argument('--output_provenance', action='store_true', default=False,
                        help='append raw detection columns and the eligible Inpaint_Mask to the CSV; these are not confidence scores')
    parser.add_argument('--output_run', action='store_true',
                        help='write a reproducible run JSON sidecar with synchronized stage timing and observable statistics')
    parser.add_argument('--label_file', default='', help='optional original-pixel ground-truth CSV for --output_run raw/final comparison')
    parser.add_argument('--tolerance', type=float, default=4.,
                        help='labeled comparison distance tolerance in 512x288 input pixels, not original video pixels')
    parser.add_argument('--traj_len', type=int, default=8, help='length of trajectory to draw on video')
    args = parser.parse_args()
    if args.label_file and not args.output_run:
        parser.error('--label_file requires --output_run')
    evaluation_protocol(args.tolerance)
    timer = StageTimer(args.output_run)
    models = {}

    num_workers = args.batch_size if args.batch_size <= 16 else 16
    video_file = args.video_file
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_range = args.video_range if args.video_range else None
    large_video = args.large_video
    out_csv_file = os.path.join(args.save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(args.save_dir, f'{video_name}.mp4')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load model
    # Load checkpoint tensors onto the selected inference device.
    # This allows CUDA-trained checkpoints to be restored on MPS or CPU systems.
    with timer.measure('model_load', device):
        tracknet_ckpt, identity = load_checkpoint(args.tracknet_file, args.output_run, device, timer)
        if args.output_run:
            models['TrackNet'] = identity
        tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
        bg_mode = tracknet_ckpt['param_dict']['bg_mode']
        tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
        tracknet.load_state_dict(tracknet_ckpt['model'])

    if args.inpaintnet_file:
        with timer.measure('model_load', device):
            inpaintnet_ckpt, identity = load_checkpoint(args.inpaintnet_file, args.output_run, device, timer)
            if args.output_run:
                models['InpaintNet'] = identity
            inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
            inpaintnet = get_model('InpaintNet').to(device)
            inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
    else:
        inpaintnet = None
        inpaintnet_seq_len = None
    settings = dict(vars(args), num_workers=0 if large_video else num_workers,
                    tracknet_seq_len=tracknet_seq_len, inpaintnet_seq_len=inpaintnet_seq_len,
                    bg_mode=bg_mode, input_size=[WIDTH, HEIGHT],
                    coordinate_suppression_threshold=COOR_TH,
                    repair_policy='same_window_raw_endpoints_v1')
    raw_pred, pred_dict, video_metadata = predict_video(
        tracknet, video_file, settings, inpaintnet=inpaintnet, timer=timer)
    with timer.measure('output_csv'):
        write_pred_csv(pred_dict, save_file=out_csv_file, save_provenance=args.output_provenance)
    if args.output_video:
        with timer.measure('output_video'):
            write_pred_video(video_file, pred_dict, save_file=out_video_file, traj_len=args.traj_len)
    if args.output_run:
        raw_metrics = final_metrics = ground_truth = None
        if args.label_file:
            with timer.measure('metrics'):
                labels = pd.read_csv(args.label_file)
                shape = (video_metadata['width'], video_metadata['height'])
                raw_metrics = metrics_from_types(evaluate_prediction_dict(raw_pred, labels, shape, args.tolerance)['Type'])
                final_metrics = metrics_from_types(evaluate_prediction_dict(pred_dict, labels, shape, args.tolerance)['Type'])
                ground_truth = file_identity(args.label_file)
        result = video_result(raw_pred, pred_dict, video_metadata, None,
                              raw_metrics, final_metrics, ground_truth)
        timing = timer.result(len(pred_dict['Frame']), includes_checkpoint_loading=True)
        result['timing'] = timing
        write_run_report(os.path.join(args.save_dir, f'{video_name}_run.json'),
                         settings, models, [result], timing, device,
                         evaluation_protocol(args.tolerance) if args.label_file else None)

    print('Done.')