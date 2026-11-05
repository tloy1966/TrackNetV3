import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Shuttlecock_Trajectory_Dataset, data_dir
from utils.general import *
from utils.inpaint import (
    format_inpaint_predictions, generate_inpaint_mask,
    get_ensemble_weight, inpaint_batch, inpaint_trajectory,
)
from utils.metric import *
from utils.runtime import (
    StageTimer, evaluation_protocol, file_identity, load_checkpoint,
    model_device, raw_prediction, prediction_statistics, video_result, write_run_report,
)
from utils.temporal import iter_heatmap_predictions


pred_types = PRED_TYPES
pred_types_map = {pred_type: i for i, pred_type in enumerate(pred_types)}
inpaintnet_eval_types = ['inpaint', 'reconstruct', 'baseline']


def predict_location(heatmap):
    """ Get coordinates from the heatmap.

        Args:
            heatmap (numpy.ndarray): A single heatmap with shape (H, W)

        Returns:
            x, y, w, h (Tuple[int, int, int, int]): bounding box of the the bounding box with max area
    """
    if np.amax(heatmap) == 0:
        # No respond in heatmap
        return 0, 0, 0, 0
    else:
        # Find all respond area in the heapmap
        (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]

        # Find largest area amoung all contours
        max_area_idx = 0
        max_area = rects[0][2] * rects[0][3]
        for i in range(1, len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        x, y, w, h = rects[max_area_idx]

        return x, y, w, h

def evaluate(indices, y_true=None, y_pred=None, c_true=None, c_pred=None, tolerance=4., img_scaler=(1, 1), output_bbox=False, output_gt=False, heatmap_threshold=0.5):
    """ Predict and output the result of each frame.

        Args:
            indices (torch.Tensor) - Indices with shape (N, L, 2)
            y_true (torch.Tensor, optional) - Ground-truth heatmap sequences with shape (N, L, H, W)
            y_pred (torch.Tensor, optional) - Predicted heatmap sequences with shape (N, L, H, W)
            c_true (torch.Tensor, optional) - Ground-truth coordinate sequences with shape (N, L, 2)
            c_pred (torch.Tensor, optional) - Predicted coordinate sequences with shape (N, L, 2)
            tolerance (float) - Tolerance for FP1
            img_scaler (Tuple[float, float]) - Scaler of input image size to original image size
            output_bbox (bool) - Whether to output detection result
            heatmap_threshold (float) - Finite [0,1] strict heatmap decoding threshold
        
        Returns:
            pred_dict (Dict) - Prediction result
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[], 'BBox': [], 'Confidence':[]}}
    """
    
    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[], 'BBox': [], 'Confidence':[], 'X_GT':[], 'Y_GT':[], 'Visibility_GT':[]}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy().tolist() if torch.is_tensor(indices) else np.asarray(indices).tolist()

    # Transform input for heatmap prediction
    if y_true is not None and y_pred is not None:
        assert c_true is None and c_pred is None, 'Invalid input'
        y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_true = to_img_format(y_true) # (N, L, H, W)
        y_pred = to_img_format(y_pred) # (N, L, H, W)
        h_pred = y_pred > validate_heatmap_threshold(heatmap_threshold)
    
    # Transform input for coordinate prediction
    if c_true is not None and c_pred is not None:
        assert y_true is None and y_pred is None, 'Invalid input'
        assert output_bbox == False, 'Coordinate prediction cannot output detection'
        c_true = c_true.detach().cpu().numpy().copy() if torch.is_tensor(c_true) else np.array(c_true, copy=True)
        c_pred = c_pred.detach().cpu().numpy().copy() if torch.is_tensor(c_pred) else np.array(c_pred, copy=True)
        c_true[..., 0] = c_true[..., 0] * WIDTH
        c_true[..., 1] = c_true[..., 1] * HEIGHT
        c_pred[..., 0] = c_pred[..., 0] * WIDTH
        c_pred[..., 1] = c_pred[..., 1] * HEIGHT

    for n in range(batch_size):
        prev_d_i = [-1, -1] # for ignoring the same frame in sequence
        for f in range(seq_len):
            d_i = indices[n][f]
            if d_i != prev_d_i:
                if c_true is not None and c_pred is not None:
                    # Predict from coordinate
                    c_t = c_true[n][f]
                    c_p = c_pred[n][f]
                    cx_true, cy_true = int(c_t[0]), int(c_t[1])
                    cx_pred, cy_pred = int(c_p[0]), int(c_p[1])
                    vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                    if np.amax(c_p) == 0 and np.amax(c_t) == 0:
                        # True Negative: prediction is no ball, and ground truth is no ball
                        pred_dict['Type'].append(pred_types_map['TN'])
                    elif np.amax(c_p) > 0 and np.amax(c_t) == 0:
                        # False Positive 2: prediction is ball existing, but ground truth is no ball
                        pred_dict['Type'].append(pred_types_map['FP2'])
                    elif np.amax(c_p) == 0 and np.amax(c_t) > 0:
                        # False Negative: prediction is no ball, but ground truth is ball existing
                        pred_dict['Type'].append(pred_types_map['FN'])
                    elif np.amax(c_p) > 0 and np.amax(c_t) > 0:
                        # Both prediction and ground truth are ball existing
                        dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                        if dist > tolerance:
                            # False Positive 1: prediction is ball existing, but is too far from ground truth
                            pred_dict['Type'].append(pred_types_map['FP1'])
                        else:
                            # True Positive
                            pred_dict['Type'].append(pred_types_map['TP'])
                    else:
                        raise ValueError(f'Invalid input: {c_p}, {c_t}')
                elif y_true is not None and y_pred is not None:
                    # Predict from heatmap
                    y_t = y_true[n][f]
                    y_p = y_pred[n][f]
                    h_p = h_pred[n][f]
                    bbox_true = predict_location(to_img(y_t))
                    cx_true, cy_true = int(bbox_true[0]+bbox_true[2]/2), int(bbox_true[1]+bbox_true[3]/2)
                    bbox_pred = predict_location(to_img(h_p))
                    cx_pred, cy_pred = int(bbox_pred[0]+bbox_pred[2]/2), int(bbox_pred[1]+bbox_pred[3]/2)
                    if np.amax(bbox_pred) > 0:
                        conf = np.amax(y_p[bbox_pred[1]:bbox_pred[1]+bbox_pred[3], bbox_pred[0]:bbox_pred[0]+bbox_pred[2]])
                    else:
                        conf = 0.
                    vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                    if np.amax(h_p) == 0 and np.amax(y_t) == 0:
                        # True Negative: prediction is no ball, and ground truth is no ball
                        pred_dict['Type'].append(pred_types_map['TN'])
                    elif np.amax(h_p) > 0 and np.amax(y_t) == 0:
                        # False Positive 2: prediction is ball existing, but ground truth is no ball
                        pred_dict['Type'].append(pred_types_map['FP2'])
                    elif np.amax(h_p) == 0 and np.amax(y_t) > 0:
                        # False Negative: prediction is no ball, but ground truth is ball existing
                        pred_dict['Type'].append(pred_types_map['FN'])
                    elif np.amax(h_p) > 0 and np.amax(y_t) > 0:
                        # Both prediction and ground truth are ball existing
                        # Find center coordinate of the contour with max area as prediction
                        dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                        if dist > tolerance:
                            # False Positive 1: prediction is ball existing, but is too far from ground truth
                            pred_dict['Type'].append(pred_types_map['FP1'])
                        else:
                            # True Positive
                            pred_dict['Type'].append(pred_types_map['TP'])
                    else:
                        raise ValueError('Invalid input')
                else:
                    raise ValueError('Invalid input')
                pred_dict['Frame'].append(int(d_i[1]))
                pred_dict['X'].append(int(cx_pred*img_scaler[0]))
                pred_dict['Y'].append(int(cy_pred*img_scaler[1]))
                pred_dict['Visibility'].append(vis_pred)

                if output_bbox:
                    pred_dict['BBox'].append([int(bbox_pred[0]*img_scaler[0]), int(bbox_pred[1]*img_scaler[1]), int(bbox_pred[2]*img_scaler[0]), int(bbox_pred[3]*img_scaler[1])])
                    pred_dict['Confidence'].append(float(conf))

                if output_gt:
                    vis_gt = 0 if cx_true == 0 and cy_true == 0 else 1
                    pred_dict['X_GT'].append(int(cx_true*img_scaler[0]))
                    pred_dict['Y_GT'].append(int(cy_true*img_scaler[1]))
                    pred_dict['Visibility_GT'].append(vis_gt)
                
                prev_d_i = d_i
            else:
                break

    if not output_bbox:
        del pred_dict['BBox']
        del pred_dict['Confidence']

    if not output_gt:
        del pred_dict['X_GT']
        del pred_dict['Y_GT']
        del pred_dict['Visibility_GT']
    
    return pred_dict


def evaluate_prediction_dict(prediction, labels, img_shape, tolerance=4.):
    """Compare original-pixel CSV predictions using the existing input-pixel evaluator."""
    evaluation_protocol(tolerance)
    prediction_statistics(prediction)
    required = {'Frame', 'X', 'Y', 'Visibility'}
    if not required.issubset(labels.columns):
        raise ValueError('Ground truth must contain Frame, X, Y, and Visibility')
    if labels['Frame'].duplicated().any() or set(labels['Frame']) != set(prediction['Frame']):
        raise ValueError('Ground truth must match output Frame IDs exactly, without duplicates')
    labels = labels.set_index('Frame').loc[prediction['Frame']]
    if not np.isin(labels['Visibility'], [0, 1]).all():
        raise ValueError('Ground-truth visibility must contain only 0 or 1')
    predicted = np.column_stack((prediction['X'], prediction['Y'])).astype(np.float64)
    truth = labels[['X', 'Y']].to_numpy(dtype=np.float64)
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError('Prediction and ground-truth coordinates must be finite')
    if len(prediction['Frame']) == 0:
        return {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Type': []}
    if len(img_shape) != 2 or not all(math.isfinite(value) and value > 0 for value in img_shape):
        raise ValueError('Image shape must contain positive, finite width and height')
    predicted[np.asarray(prediction['Visibility']) == 0] = 0.
    truth[labels['Visibility'].to_numpy() == 0] = 0.
    indices = np.column_stack((np.zeros(len(predicted), dtype=np.int64),
                               np.asarray(prediction['Frame'], dtype=np.int64)))
    return evaluate(indices[:, None, :], c_true=(truth / img_shape)[:, None, :],
                    c_pred=(predicted / img_shape)[:, None, :], tolerance=tolerance)


def linear_interp(target, inpaint_mask):
    """Interpolate bounded masked runs without extrapolating either endpoint."""
    if len(target) != len(inpaint_mask):
        raise ValueError('Target and inpaint mask must have equal lengths')
    if not np.isin(inpaint_mask, [0, 1]).all():
        raise ValueError('Inpaint mask must contain only 0 or 1')
    target = np.array(target, dtype=np.float64, copy=True)
    start = 0
    while start < len(inpaint_mask):
        if inpaint_mask[start] == 0:
            start += 1
            continue
        end = start
        while end < len(inpaint_mask) and inpaint_mask[end] == 1:
            end += 1
        if start > 0 and end < len(inpaint_mask):
            target[start:end] = np.interp(np.arange(start, end), [start - 1, end],
                                          [target[start - 1], target[end]])
        start = end
    return target

# Only for training evaluation, won't save the result
def get_eval_res(pred_dict):
    """ Parse prediction result and get evaluation result.

        Args:
            pred_dict (Dict): Prediction result
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[]}
        
        Returns:
            res (numpy.ndarray): Evaluation result
                Format: np.array([TP, TN, FP1, FP2, FN])
    """

    type_res = np.array(pred_dict['Type'])
    res = np.zeros(5)
    for pred_type in pred_types:
        res[pred_types_map[pred_type]] += int((type_res == pred_types_map[pred_type]).sum())

    return res

def eval_tracknet(model, data_loader, param_dict):
    """ Evaluate TrackNet model.

        Args:
            model (nn.Module): TrackNet model
            data_loader (torch.utils.data.DataLoader): DataLoader for evaluation
            param_dict (Dict): Parameters
                param_dict['verbose'] (bool): Whether to show progress bar
                param_dict['tolerance'] (int): Tolerance for FP1
            
        Returns:
            (float): Average loss
            res_dict (Dict): Evaluation result
                Format:{'TP': TP, 'TN': TN,
                        'FP1': FP1, 'FP2': FP2, 'FN': FN,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'miss_rate': miss_rate}
    """

    model.eval()
    losses = []
    confusion_matrix = np.zeros(5) # TP, TN, FP1, FP2, FN
    if param_dict['verbose']:
        data_prob = tqdm(data_loader)
    else:
        data_prob = data_loader
    
    for step, (i, x, y, _, _) in enumerate(data_prob):
        x, y = x.float().cuda(), y.float().cuda()
        with torch.no_grad():
            y_pred = model(x)

        loss = WBCELoss(y_pred, y)
        losses.append(loss.item())

        pred_dict = evaluate(i, y_true=y, y_pred=y_pred, tolerance=param_dict['tolerance'],
                             heatmap_threshold=param_dict.get('heatmap_threshold', 0.5))
        confusion_matrix += get_eval_res(pred_dict)
        
        if param_dict['verbose']:
            TP, TN, FP1, FP2, FN = confusion_matrix
            data_prob.set_description(f'Evaluation')
            data_prob.set_postfix(TP=TP, TN=TN, FP1=FP1, FP2=FP2, FN=FN)
    
    TP, TN, FP1, FP2, FN = confusion_matrix
    accuracy, precision, recall, f1, miss_rate = get_metric(TP, TN, FP1, FP2, FN)
    res_dict = {'TP': TP, 'TN': TN,
                'FP1': FP1, 'FP2': FP2, 'FN': FN,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'miss_rate': miss_rate}
    
    return float(np.mean(losses)), res_dict

def eval_inpaintnet(model, data_loader, param_dict):
    """ Evaluate InpaintNet with the same observation-bounded window policy as inference.

        Args:
            model (nn.Module): InpaintNet model
            data_loader (torch.utils.data.DataLoader): DataLoader for evaluation
            param_dict (Dict): Parameters
                param_dict['verbose'] (bool): Whether to show progress bar
                param_dict['tolerance'] (int): Tolerance for FP1
            
        Returns:
            (float): Average loss
            res_dict (Dict): Evaluation result
                Format:{'TP': TP, 'TN': TN,
                        'FP1': FP1, 'FP2': FP2, 'FN': FN,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'miss_rate': miss_rate}
    """

    model.eval()
    losses = []
    confusion_matrix = {eval_type: np.zeros(5) for eval_type in inpaintnet_eval_types} # TP, TN, FP1, FP2, FN
    if param_dict['verbose']:
        data_prob = tqdm(data_loader)
    else:
        data_prob = data_loader

    for step, (i, coor_pred, coor, pred_vis, _, inpaint_mask) in enumerate(data_prob):
        coor_pred, coor, inpaint_mask = coor_pred.float().cuda(), coor.float().cuda(), inpaint_mask.float().cuda()
        coor_pred = coor_pred.masked_fill(~pred_vis.cuda().bool(), 0.)
        
        with torch.no_grad():
            coor_inpaint, repair_mask = inpaint_batch(
                model, coor_pred, inpaint_mask, indices=i, coor_th=COOR_TH)
            
            loss = nn.MSELoss()(coor_inpaint * repair_mask, coor * repair_mask)
            losses.append(loss.item())
        
        for eval_type in inpaintnet_eval_types:
            if eval_type == 'inpaint':
                pred_dict = evaluate(i, c_true=coor, c_pred=coor_inpaint, tolerance=param_dict['tolerance'])
            elif eval_type == 'reconstruct':
                pred_dict = evaluate(i, c_true=coor_pred, c_pred=coor_inpaint, tolerance=param_dict['tolerance'])
            elif eval_type == 'baseline':
                pred_dict = evaluate(i, c_true=coor, c_pred=coor_pred, tolerance=param_dict['tolerance'])
            else:
                raise ValueError('Invalid eval_type')
            confusion_matrix[eval_type] += get_eval_res(pred_dict)
        
        if param_dict['verbose']:
            TP, TN, FP1, FP2, FN = confusion_matrix['inpaint']
            data_prob.set_description(f'Evaluation')
            data_prob.set_postfix(TP=TP, TN=TN, FP1=FP1, FP2=FP2, FN=FN)
    
    res_dict = {}
    for eval_type in inpaintnet_eval_types:
        TP, TN, FP1, FP2, FN = confusion_matrix[eval_type]
        accuracy, precision, recall, f1, miss_rate = get_metric(TP, TN, FP1, FP2, FN)
        res_dict[eval_type] = {'TP': TP, 'TN': TN,
                               'FP1': FP1, 'FP2': FP2, 'FN': FN,
                               'accuracy': accuracy,
                               'precision': precision,
                               'recall': recall,
                               'f1': f1,
                               'miss_rate': miss_rate}
    
    return float(np.mean(losses)), res_dict

# For testing evaluation
def get_coco_res(pred_dict, drop=False):
    """ Parse prediction result and get evaluation result.

        Args:
            pred_dict (Dict): Prediction result
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[], 'BBox': [], 'Confidence': []}
            drop (bool): Whether to drop the frames in the drop frame range

        Returns:
            res_dict (Dict): COCO format evaluation result
                Format: [{'image_id': int, 'category_id': int, 'bbox': [x, y, w, h], 'score': float}, ...]
    """
    sample_count = 0
    res_list = []
    if drop and pred_dict:
        with open(os.path.join(data_dir, 'drop_frame.json')) as source:
            drop_frame_dict = json.load(source)
        start_f, end_f = drop_frame_dict['start'], drop_frame_dict['end']
    for rally_key, pred in pred_dict.items():
        if drop:
            pred = {key: values[start_f[rally_key]:end_f[rally_key]] for key, values in pred.items()}
        
        for i in range(len(pred['Frame'])):
            if pred['Visibility'][i] > 0:
                res_list.append({'id': sample_count,
                                 'image_id': sample_count,
                                 'category_id': 1,
                                 'bbox': pred['BBox'][i],
                                 'score': pred['Confidence'][i],
                                 'ignore': 0,
                                 'area': pred['BBox'][i][2]*pred['BBox'][i][3],
                                 'segmentation': [],
                                 'iscrowd': 0})
            sample_count += 1
    
    return res_list

def get_test_res(pred_dict, drop=False, type_key='Type'):
    """ Parse prediction result and get evaluation result.

        Args:
            pred_dict (Dict): Prediction result
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[]}
            drop (bool): Whether to drop the frames in the drop frame range

        Returns:
            res_dict (Dict): Evaluation result
                Format: {'TP': TP, 'TN': TN,
                         'FP1': FP1, 'FP2': FP2, 'FN': FN,
                         'accuracy': accuracy,
                         'precision': precision,
                         'recall': recall,
                         'f1': f1,
                         'miss_rate': miss_rate}
    """

    res_dict = {pred_type: 0 for pred_type in pred_types}
    if drop and pred_dict:
        with open(os.path.join(data_dir, 'drop_frame.json')) as source:
            drop_frame_dict = json.load(source)
        start_f, end_f = drop_frame_dict['start'], drop_frame_dict['end']
    for rally_key, pred in pred_dict.items():
        if drop:
            type_res = np.array(pred[type_key])[start_f[rally_key]:end_f[rally_key]]
        else:
            type_res = np.array(pred[type_key])
        
        # Calculate metrics
        for pred_type in pred_types:
            res_dict[pred_type] += int((type_res == pred_types_map[pred_type]).sum())
    
    TP, TN, FP1, FP2, FN = res_dict['TP'], res_dict['TN'], res_dict['FP1'], res_dict['FP2'], res_dict['FN']
    accuracy, precision, recall, f1, miss_rate = get_metric(TP, TN, FP1, FP2, FN)
    res_dict = {'TP': TP, 'TN': TN,
                'FP1': FP1, 'FP2': FP2, 'FN': FN,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'miss_rate': miss_rate}
    
    return res_dict

def rally_run_result(pred, info, param_dict, rally_key, drop=False):
    final = dict(pred)
    if info['repair_method'] == 'none':
        final['Inpaint_Mask'] = [0] * len(final['Frame'])
    evaluated = pred
    metadata = {key: value for key, value in info.items() if key != 'ground_truth'}
    if drop:
        with open(os.path.join(data_dir, 'drop_frame.json')) as source:
            ranges = json.load(source)
        start, end = ranges['start'][rally_key], ranges['end'][rally_key]
        evaluated = {key: pred[key][start:end] for key in ('Type', 'Type_Raw')}
        metadata['evaluation_rows'] = {'start_inclusive': start, 'end_exclusive': end}
    raw_metrics = get_test_res({rally_key: evaluated}, type_key='Type_Raw')
    final_metrics = get_test_res({rally_key: evaluated})
    result = video_result(raw_prediction(final), final, metadata, None,
                          raw_metrics, final_metrics, info['ground_truth'])
    result['evaluated_frames'] = sum(final_metrics[key] for key in pred_types)
    result['evaluation_protocol'] = evaluation_protocol(param_dict['tolerance'])
    return result


def test(model, split, param_dict, save_inpaint_mask=False, linear_interp=False,
         timer=None, run_results=None, run_models=None):
    """ Test model on all the rallies in the split.

        Args:
            model (nn.Module): TrackNet model
            split (str): Split for testing
                Choices: 'train', 'val', 'test'
            param_dict (Dict): Parameters
            save_inpaint_mask (bool): Whether to save inpaint mask to '{data_dir}/match{match_id}/predicted_csv/{rally_id}_ball.csv'

        Returns:
            pred_dict (Dict): Evaluation result
                Format: {'{match_id}_{rally_id}': {
                            'TP': TP, 'TN': TN,
                            'FP1': FP1, 'FP2': FP2, 'FN': FN,
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'miss_rate': miss_rate}, ...
                        }
    """

    # Rally-based test
    pred_dict = {}
    timer = timer if timer is not None else StageTimer()
    rally_dirs = get_rally_dirs(data_dir, split)
    rally_dirs = [os.path.join(data_dir, rally_dir) for rally_dir in rally_dirs]
    if param_dict['debug']:
        rally_dirs = rally_dirs[:1]
    
    for rally_dir in rally_dirs:
        # Parse rally directory to form rally key
        file_format_str = os.path.join('{}', 'frame', '{}')
        match_dir, rally_id = parse.parse(file_format_str, rally_dir)
        match_id = match_dir.split('match')[-1]
        rally_key = f'{match_id}_{rally_id}'

        video_timer = StageTimer(run_results is not None)
        run_info = {} if run_results is not None else None
        if linear_interp:
            tmp_pred = test_rally_linear(model, rally_dir, param_dict, timer=video_timer, run_info=run_info)
        else:
            tmp_pred = test_rally(model, rally_dir, param_dict, save_inpaint_mask=save_inpaint_mask,
                                  timer=video_timer, run_info=run_info)
        pred_dict[rally_key] = tmp_pred

        if save_inpaint_mask:
            if not os.path.exists(os.path.join(match_dir, 'predicted_csv')):
                os.makedirs(os.path.join(match_dir, 'predicted_csv'))
            csv_file = os.path.join(match_dir, 'predicted_csv',f'{rally_id}_ball.csv')
            with video_timer.measure('output_csv'):
                write_pred_csv(tmp_pred, save_file=csv_file, save_inpaint_mask=save_inpaint_mask)
        if run_results is not None:
            with video_timer.measure('metrics'):
                result = rally_run_result(tmp_pred, run_info, param_dict, rally_key,
                                          drop=split == 'test' and not save_inpaint_mask)
            if save_inpaint_mask:
                result['prediction_csv'] = file_identity(csv_file)
            result['timing'] = video_timer.result(len(tmp_pred['Frame']))
            if save_inpaint_mask:
                write_run_report(csv_file + '.run.json', param_dict, run_models, [result],
                                 result['timing'], model_device(model[0]),
                                 evaluation_protocol(param_dict['tolerance']))
            run_results.append(result)
        timer.merge(video_timer)
    
    return pred_dict

def test_rally(model, rally_dir, param_dict, save_inpaint_mask=False, timer=None, run_info=None):
    """ Test model on a single rally.

        Args:
            model (Tuple[nn.Module, nn.Module]): TrackNet model
            rally_dir (str): Directory of the rally
            param_dict (Dict): Parameters
                param_dict['eval_mode'] (str): Mode of temporal ensemble
                param_dict['tolerance'] (int): Tolerance for FP1
                param_dict['bg_mode'] (str): Mode of background
                param_dict['batch_size'] (int): Batch size
                param_dict['num_workers'] (int): Number of workers
                param_dict['tracknet_seq_len'] (int): Length of input sequence for TrackNet
                param_dict['inpaintnet_seq_len'] (int): Length of input sequence for InpaintNet

        Returns:
            pred_dict (Dict): Evaluation result
                Format: {'TP': TP, 'TN': TN,
                         'FP1': FP1, 'FP2': FP2, 'FN': FN,
                         'accuracy': accuracy,
                         'precision': precision,
                         'recall': recall,
                         'f1': f1,
                         'miss_rate': miss_rate}
    """

    tracknet, inpaintnet = model
    timer = timer if timer is not None else StageTimer()
    with Image.open(os.path.join(rally_dir, '0.png')) as image:
        w, h = image.size
    if run_info is not None:
        run_info.update(path=rally_dir, width=w, height=h, fps=None,
                        input_source='rally_images' if tracknet is not None else 'cached_coordinates',
                        heatmap_threshold_applied=param_dict.get('heatmap_threshold', 0.5) if tracknet is not None else None)
        match_dir, rally_id = parse.parse(os.path.join('{}', 'frame', '{}'), rally_dir)
        video_file = os.path.join(match_dir, 'video', f'{rally_id}.mp4')
        run_info['video_file'] = {'path': video_file, 'available': os.path.isfile(video_file)}
        if os.path.isfile(video_file):
            with timer.measure('video_probe'):
                cap = cv2.VideoCapture(video_file)
                try:
                    opened = cap.isOpened()
                    run_info['video_file']['probe_success'] = bool(opened)
                    if opened:
                        fps = float(cap.get(cv2.CAP_PROP_FPS))
                        run_info['fps'] = fps if math.isfinite(fps) and fps > 0 else None
                        run_info['declared_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                finally:
                    cap.release()
    if save_inpaint_mask:
        w_scaler, h_scaler = 1., 1.
    else:
        w_scaler, h_scaler = w / WIDTH, h / HEIGHT

    # Test on TrackNet
    if inpaintnet is None:
        tracknet.eval()
        seq_len = param_dict['tracknet_seq_len']
        tracknet_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[], 'BBox': [], 'Confidence':[], 'X_GT':[], 'Y_GT':[], 'Visibility_GT':[]}

        nonoverlap = param_dict['eval_mode'] == 'nonoverlap'
        step = seq_len if nonoverlap or get_num_frames(rally_dir) < seq_len else 1
        with timer.measure('input_setup'):
            dataset = Shuttlecock_Trajectory_Dataset(
                root_dir=data_dir, seq_len=seq_len, sliding_step=step, data_mode='heatmap',
                bg_mode=param_dict['bg_mode'], rally_dir=rally_dir, padding=step == seq_len)
            data_loader = DataLoader(dataset, batch_size=param_dict['batch_size'], shuffle=False, num_workers=param_dict['num_workers'], drop_last=False)
        data_prob = tqdm(data_loader) if param_dict['verbose'] else data_loader
        for i, y, y_pred in iter_heatmap_predictions(
                tracknet, data_prob, param_dict['eval_mode'], model_device(tracknet), timer):
            with timer.measure('postprocess'):
                tmp_pred = evaluate(i, y_true=y, y_pred=y_pred,
                                    tolerance=param_dict['tolerance'], img_scaler=(w_scaler, h_scaler),
                                    output_bbox=param_dict['output_bbox'], output_gt=param_dict['output_gt'],
                                    heatmap_threshold=param_dict.get('heatmap_threshold', 0.5))
                for key in tmp_pred.keys():
                    tracknet_pred_dict[key].extend(tmp_pred[key])
        
        tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=30)
        if param_dict.get('output_provenance', False) and not save_inpaint_mask:
            tracknet_pred_dict.update(format_inpaint_predictions(
                tracknet_pred_dict, torch.zeros((len(tracknet_pred_dict['Frame']), 2)),
                torch.zeros((len(tracknet_pred_dict['Frame']), 1), dtype=torch.bool), (w, h)))
        if run_info is not None:
            tracknet_pred_dict['Type_Raw'] = list(tracknet_pred_dict['Type'])
            for key in ('X', 'Y', 'Visibility'):
                tracknet_pred_dict[f'{key}_Raw'] = list(tracknet_pred_dict[key])
            file_format_str = os.path.join('{}', 'frame', '{}')
            match_dir, rally_id = parse.parse(file_format_str, rally_dir)
            label_dir = 'corrected_csv' if 'test' in rally_dir else 'csv'
            run_info.update(processed_frames=len(tracknet_pred_dict['Frame']), raw_source='fresh_tracknet',
                            ground_truth=file_identity(os.path.join(match_dir, label_dir, f'{rally_id}_ball.csv')),
                            repair_method='none', num_workers=data_loader.num_workers)
            if param_dict['bg_mode']:
                median_file = os.path.join(match_dir, 'median.npz')
                if not os.path.isfile(median_file):
                    median_file = os.path.join(rally_dir, 'median.npz')
                run_info['background_file'] = file_identity(median_file)
        return tracknet_pred_dict
    else:
        # Test on TrackNetV3 (TrackNet + InpaintNet)
        inpaintnet.eval()
        seq_len = param_dict['inpaintnet_seq_len']
        file_format_str = os.path.join('{}', 'frame', '{}')
        match_dir, rally_id = parse.parse(file_format_str, rally_dir)
        pred_csv_file = os.path.join(match_dir, 'predicted_csv', f'{rally_id}_ball.csv')
        if tracknet is not None:
            tracking_info = {} if run_info is not None else None
            tracking = test_rally(
                (tracknet, None), rally_dir,
                dict(param_dict, output_gt=True, output_bbox=False, output_provenance=False),
                save_inpaint_mask=True, timer=timer, run_info=tracking_info)
            pred_df = pd.DataFrame({key: tracking[key] for key in
                                    ('Frame', 'X', 'Y', 'Visibility', 'X_GT', 'Y_GT', 'Visibility_GT', 'Inpaint_Mask')})
            if run_info is not None:
                run_info.update(tracking_info)
        else:
            with timer.measure('input_setup'):
                pred_df = pd.read_csv(pred_csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)
            if run_info is not None:
                cache_identity = file_identity(pred_csv_file)
                run_info.update(raw_source='cached_coordinates', prediction_cache=cache_identity,
                                ground_truth=dict(cache_identity, columns=['X_GT', 'Y_GT', 'Visibility_GT']),
                                requested_heatmap_threshold=param_dict.get('heatmap_threshold', 0.5),
                                cache_generation=None, num_workers=0)
                sidecar = pred_csv_file + '.run.json'
                if os.path.isfile(sidecar):
                    with open(sidecar, encoding='utf8') as source:
                        generation = json.load(source)
                    generated_video = generation['videos'][0]
                    if generated_video['prediction_csv']['sha256'] != cache_identity['sha256']:
                        raise ValueError('Cached prediction CSV no longer matches its run sidecar; regenerate it')
                    run_info['heatmap_threshold_applied'] = validate_heatmap_threshold(
                        generation['settings']['heatmap_threshold'])
                    run_info['cache_generation'] = {
                        'sidecar': file_identity(sidecar), 'models': generation['models'],
                        'heatmap_threshold': run_info['heatmap_threshold_applied'],
                    }
        raw_pred = {key: pred_df[key].tolist() for key in ('Frame', 'X', 'Y', 'Visibility')}
        coordinates = torch.tensor(pred_df[['X', 'Y']].to_numpy(), dtype=torch.float32)
        coordinates /= torch.tensor((WIDTH, HEIGHT))
        coordinates[torch.tensor(raw_pred['Visibility']) == 0] = 0.
        raw_coordinates = coordinates.clone() if run_info is not None else None
        candidates = np.asarray(generate_inpaint_mask(raw_pred, th_h=30, seq_len=seq_len), dtype=np.int64)
        cached_mask = pred_df['Inpaint_Mask'].to_numpy()
        if not np.isin(cached_mask, [0, 1]).all():
            raise ValueError('Stored Inpaint_Mask must contain only 0 or 1')
        candidates &= cached_mask.astype(np.int64)
        indices = torch.stack((torch.zeros(len(pred_df), dtype=torch.long),
                               torch.tensor(raw_pred['Frame'], dtype=torch.long)), dim=1)
        with timer.measure('inpainting', model_device(inpaintnet)):
            coordinates, repair_mask = inpaint_trajectory(
                inpaintnet, coordinates, torch.tensor(candidates, dtype=torch.float32).reshape(-1, 1),
                seq_len, eval_mode=param_dict['eval_mode'], batch_size=param_dict['batch_size'],
                device=model_device(inpaintnet), indices=indices, coor_th=COOR_TH, timer=timer)

        ground_truth = torch.tensor(pred_df[['X_GT', 'Y_GT']].to_numpy(), dtype=torch.float32)
        ground_truth /= torch.tensor((WIDTH, HEIGHT))
        ground_truth[torch.tensor(pred_df['Visibility_GT'].to_numpy()) == 0] = 0.
        with timer.measure('postprocess'):
            inpaintnet_pred_dict = evaluate(
                indices[:, None, :], c_true=ground_truth[:, None, :], c_pred=coordinates[:, None, :],
                tolerance=param_dict['tolerance'], img_scaler=(w_scaler, h_scaler))
        if run_info is not None:
            inpaintnet_pred_dict['Type_Raw'] = evaluate(
                indices[:, None, :], c_true=ground_truth[:, None, :], c_pred=raw_coordinates[:, None, :],
                tolerance=param_dict['tolerance'])['Type']
            run_info.update(processed_frames=len(raw_pred['Frame']), repair_method='inpaintnet')
        raw_pred['X'] = [int(x * w_scaler) for x in raw_pred['X']]
        raw_pred['Y'] = [int(y * h_scaler) for y in raw_pred['Y']]
        img_shape = (WIDTH, HEIGHT) if save_inpaint_mask else (w, h)
        inpaintnet_pred_dict.update(format_inpaint_predictions(
            raw_pred, coordinates, repair_mask, img_shape))
        return inpaintnet_pred_dict
        
def test_rally_linear(model, rally_dir, param_dict, timer=None, run_info=None):
    tracknet, _ = model
    with Image.open(os.path.join(rally_dir, '0.png')) as image:
        w, h = image.size
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT

    timer = timer if timer is not None else StageTimer()
    tracknet_pred_dict = test_rally(
        (tracknet, None), rally_dir,
        dict(param_dict, output_gt=True, output_bbox=False, output_provenance=False),
        save_inpaint_mask=True, timer=timer, run_info=run_info)
    
    context_len = param_dict.get('inpaintnet_seq_len', param_dict['tracknet_seq_len'])
    tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(
        tracknet_pred_dict, th_h=30, seq_len=context_len)

    
    with timer.measure('postprocess'):
        mask = tracknet_pred_dict['Inpaint_Mask']
        coordinates = np.column_stack((linear_interp(tracknet_pred_dict['X'], mask) / WIDTH,
                                       linear_interp(tracknet_pred_dict['Y'], mask) / HEIGHT))
        truth = np.column_stack((np.asarray(tracknet_pred_dict['X_GT']) / WIDTH,
                                 np.asarray(tracknet_pred_dict['Y_GT']) / HEIGHT))
        indices = np.column_stack((np.zeros(len(coordinates), dtype=np.int64),
                                   np.asarray(tracknet_pred_dict['Frame'], dtype=np.int64)))
        inpaintnet_pred_dict = evaluate(indices[:, None, :], c_true=truth[:, None, :],
                                        c_pred=coordinates[:, None, :], tolerance=param_dict['tolerance'],
                                        img_scaler=(w_scaler, h_scaler))
    raw_pred = {key: list(tracknet_pred_dict[key]) for key in ('Frame', 'X', 'Y', 'Visibility')}
    raw_pred['X'] = [int(x * w_scaler) for x in raw_pred['X']]
    raw_pred['Y'] = [int(y * h_scaler) for y in raw_pred['Y']]
    inpaintnet_pred_dict.update(format_inpaint_predictions(
        raw_pred, coordinates,
        np.asarray(tracknet_pred_dict['Inpaint_Mask']).reshape(-1, 1), (w, h)))
    
    if run_info is not None:
        inpaintnet_pred_dict['Type_Raw'] = tracknet_pred_dict['Type']
        run_info['repair_method'] = 'bounded_linear_interpolation'
    return inpaintnet_pred_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracknet_file', type=str, help='file path of the TrackNet model checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='file path of the InpaintNet model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='dataset split for testing')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for testing')
    parser.add_argument('--tolerance', type=float, default=4, help='difference tolerance of center distance between prediction and ground truth in input size')
    parser.add_argument('--heatmap_threshold', type=validate_heatmap_threshold, default=0.5,
                        help='strict heatmap > threshold in [0,1] for fresh heatmaps; cached coordinates are unchanged')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--video_file', type=str, default='', help='file path of the video with label (must in dataset directory with same data_dir)')
    parser.add_argument('--output_pred', action='store_true', default=False, help='whether to output detail prediction result for error analysis')
    parser.add_argument('--output_provenance', action='store_true', default=False,
                        help='append raw detection columns and eligible Inpaint_Mask to video-evaluation CSV output; include --output_pred for dataset details')
    parser.add_argument('--output_run', action='store_true',
                        help='write reproducible raw/final metrics, checkpoint identities, protocol, and synchronized stage timing')
    parser.add_argument('--output_bbox', action='store_true', default=False, help='whether to output coco format bbox prediction result for mAP evaluation')
    parser.add_argument('--save_dir', type=str, default='output', help='directory to save the evaluation result')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--linear_interp', action='store_true', default=False,
                        help='evaluate bounded linear repair; maximum gap is context length minus two (InpaintNet length, or TrackNet length without InpaintNet)')
    args = parser.parse_args()
    if not args.tracknet_file and not args.inpaintnet_file:
        parser.error('provide --tracknet_file or --inpaintnet_file')
    if args.linear_interp and not args.tracknet_file:
        parser.error('--linear_interp requires --tracknet_file')
    evaluation_protocol(args.tolerance)
    timer = StageTimer(args.output_run)
    run_results = [] if args.output_run else None
    run_models = {}

    param_dict = vars(args)
    param_dict['num_workers'] = args.batch_size if args.batch_size <= 16 else 16
    param_dict['output_bbox'] = args.output_bbox
    param_dict['output_gt'] = False
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load parameter
    print(f'Loading checkpoint...')
    if args.tracknet_file:
        with timer.measure('model_load'):
            tracknet_ckpt, identity = load_checkpoint(args.tracknet_file, args.output_run, timer=timer)
            if args.output_run:
                run_models['TrackNet'] = identity
            param_dict['tracknet_seq_len'] = tracknet_ckpt['param_dict']['seq_len']
            param_dict['bg_mode'] = tracknet_ckpt['param_dict']['bg_mode']
            tracknet = get_model('TrackNet', seq_len=param_dict['tracknet_seq_len'], bg_mode=param_dict['bg_mode']).cuda()
            tracknet.load_state_dict(tracknet_ckpt['model'])
            if timer.enabled:
                timer.synchronize(model_device(tracknet))
        model = (tracknet, None)
    else:
        tracknet = None
    
    if args.inpaintnet_file:
        with timer.measure('model_load'):
            inpaintnet_ckpt, identity = load_checkpoint(args.inpaintnet_file, args.output_run, timer=timer)
            if args.output_run:
                run_models['InpaintNet'] = identity
            param_dict['inpaintnet_seq_len'] = inpaintnet_ckpt['param_dict']['seq_len']
            inpaintnet = get_model('InpaintNet').cuda()
            inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
            if timer.enabled:
                timer.synchronize(model_device(inpaintnet))
        model = (tracknet, inpaintnet)
        if tracknet is None:
            print('Using cached TrackNet coordinates; the requested heatmap threshold is not applied to cached coordinates.')

    if args.video_file:
        # Evaluation on video
        print(f'Test on video {args.video_file} ...')
        file_format_str = os.path.join('{}', 'video', '{}.mp4')
        match_dir, rally_id = parse.parse(file_format_str, args.video_file)
        rally_dir = os.path.join(match_dir, 'frame', rally_id)

        # Load label
        csv_file = os.path.join(match_dir, 'corrected_csv', f'{rally_id}_ball.csv') if 'test' in rally_dir else os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
        assert os.path.exists(csv_file), f'{csv_file} does not exist.'
        label_df = pd.read_csv(csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)

        # Predict label
        run_info = {} if args.output_run else None
        evaluate_rally = test_rally_linear if args.linear_interp else test_rally
        pred_dict = evaluate_rally(model, rally_dir, param_dict, timer=timer, run_info=run_info)

        # Write results
        out_video_file = os.path.join(args.save_dir, f'{rally_id}.mp4')
        out_csv_file = os.path.join(args.save_dir, f'{rally_id}_ball.csv')
        with timer.measure('output_video'):
            write_pred_video(args.video_file, pred_dict, label_df=label_df, save_file=out_video_file)
        with timer.measure('output_csv'):
            write_pred_csv(pred_dict, save_file=out_csv_file, save_provenance=args.output_provenance)
        if args.output_run:
            with timer.measure('metrics'):
                result = rally_run_result(pred_dict, run_info, param_dict, rally_id)
            result['timing'] = timer.result(len(pred_dict['Frame']), includes_checkpoint_loading=True)
            run_results.append(result)
            report_file = os.path.join(args.save_dir, f'{rally_id}_run.json')
    else:
        # Evaluation on dataset
        eval_analysis_file = os.path.join(args.save_dir, f'{args.split}_eval_analysis_{args.eval_mode}.json') # for error analysis interface
        eval_res_file = os.path.join(args.save_dir, f'{args.split}_eval_res_{args.eval_mode}.json')

        start_time = time.time()
        print(f'Split: {args.split}')
        print(f'Evaluation mode: {args.eval_mode}')
        print(f'Tolerance Value: {args.tolerance}')
        
        pred_dict = test(model, args.split, param_dict, linear_interp=args.linear_interp,
                         timer=timer, run_results=run_results, run_models=run_models)
        if args.split == 'test':
            # Drop samples which is not in the effective trajectory
            res_dict = get_test_res(pred_dict, drop=True)
        else:
            res_dict = get_test_res(pred_dict, drop=False)
        
        with timer.measure('output_metrics'), open(eval_res_file, 'w') as f:
            json.dump(res_dict, f, indent=2)

        if args.output_pred:
            eval_dict = dict(param_dict=param_dict, pred_dict=pred_dict)
            with timer.measure('output_predictions'), open(eval_analysis_file, 'w') as f:
                json.dump(eval_dict, f, indent=2)

        if args.output_bbox:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            coco_file = os.path.join(args.save_dir, f'{args.split}_coco_res_{args.eval_mode}.json')
            if args.split == 'test':
                dect_list = get_coco_res(pred_dict, drop=True)
            else:
                dect_list = get_coco_res(pred_dict, drop=False)
            
            mAP = {0.25:0, 0.5:0}
            coco_gt = COCO(os.path.join(data_dir, 'coco_format_gt.json'))
            coco_dt = coco_gt.loadRes(dect_list)
            for iou_th in [0.25, 0.5]:
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.params.iouThrs = [iou_th]
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                mAP[iou_th] = coco_eval.stats[0]
                
            coco_res_dict = dict(AP_25=mAP, detection=dect_list)
            with open(coco_file, 'w') as f:
                json.dump(coco_res_dict, f, indent=2)
        if args.output_run:
            report_file = os.path.join(args.save_dir, f'{args.split}_run_{args.eval_mode}.json')
    if args.output_run:
        frame_filter = (file_identity(os.path.join(data_dir, 'drop_frame.json'))
                        if not args.video_file and args.split == 'test' else None)
        original_frames = sum(result['statistics']['final']['frames'] for result in run_results)
        settings = dict(param_dict, data_dir=data_dir, input_size=[WIDTH, HEIGHT],
                        coordinate_suppression_threshold=COOR_TH,
                        repair_policy='same_window_raw_endpoints_v1')
        write_run_report(report_file, settings, run_models, run_results,
                         timer.result(original_frames, includes_checkpoint_loading=True),
                         model_device(model[0] if model[0] is not None else model[1]),
                         evaluation_protocol(args.tolerance, frame_filter))
