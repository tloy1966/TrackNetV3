import os
import torch
import argparse

from test import test
from dataset import data_dir
from utils.general import get_model
from utils.metric import validate_heatmap_threshold
from utils.runtime import StageTimer, evaluation_protocol, load_checkpoint, model_device, write_run_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracknet_file', type=str, help='TrackNet model file')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--heatmap_threshold', type=validate_heatmap_threshold, default=0.5,
                        help='strict heatmap > threshold decoding used to generate coordinates and masks')
    parser.add_argument('--output_run', action='store_true',
                        help='write cache-identifying per-CSV run sidecars and a mask-generation run JSON in the data root')
    parser.add_argument('--split_list', type=lambda splits: [s for s in splits.split(',')], default='train,val,test', help='split list')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    timer = StageTimer(args.output_run)
    run_results = [] if args.output_run else None

    # Load model
    with timer.measure('model_load'):
        ckpt, identity = load_checkpoint(args.tracknet_file, args.output_run, timer=timer)

    # Load and set parameters
    param_dict = dict(ckpt['param_dict'])
    param_dict['num_workers'] = args.batch_size
    param_dict['batch_size'] = args.batch_size
    param_dict['eval_mode'] = args.eval_mode
    param_dict['heatmap_threshold'] = args.heatmap_threshold
    param_dict['output_run'] = args.output_run
    param_dict['split_list'] = args.split_list
    param_dict['debug'] = args.debug
    param_dict['verbose'] = args.verbose
    param_dict['tracknet_seq_len'] = param_dict['seq_len']
    param_dict['output_bbox'] = False
    param_dict['output_gt'] = True
    with timer.measure('model_load'):
        tracknet = get_model(param_dict['model_name'], param_dict['seq_len'], param_dict['bg_mode']).cuda()
        tracknet.load_state_dict(ckpt['model'])
        if timer.enabled:
            timer.synchronize(model_device(tracknet))
    run_models = {'TrackNet': identity} if args.output_run else None
    
    for split in args.split_list:
        print(f'Generating predicted trajectories and inpainting masks for {split} set...')
        _ = test((tracknet, None), split, param_dict, save_inpaint_mask=True,
                 timer=timer, run_results=run_results, run_models=run_models)
    if args.output_run:
        original_frames = sum(result['statistics']['final']['frames'] for result in run_results)
        write_run_report(os.path.join(data_dir, f'mask_generation_run_{args.eval_mode}.json'),
                         param_dict, run_models, run_results,
                         timer.result(original_frames, includes_checkpoint_loading=True),
                         model_device(tracknet), evaluation_protocol(param_dict['tolerance']))