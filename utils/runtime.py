import hashlib
import json
import math
import os
import platform
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
import PIL
import torch

from utils.general import HEIGHT, WIDTH
from utils.metric import PRED_TYPES, metrics_from_counts


class StageTimer:
    """Opt-in, synchronized wall-clock stages; nested stages are not additive."""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.seconds = {}
        self.calls = {}
        self.counts = {}
        self.started = time.perf_counter() if enabled else None

    @staticmethod
    def synchronize(device):
        if device is None:
            return
        device = torch.device(device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elif device.type == 'mps':
            torch.mps.synchronize()

    @contextmanager
    def measure(self, name, device=None):
        if not self.enabled:
            yield
            return
        self.synchronize(device)
        started = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize(device)
            self.seconds[name] = self.seconds.get(name, 0.) + time.perf_counter() - started
            self.calls[name] = self.calls.get(name, 0) + 1

    def count(self, name, count=1):
        self.counts[name] = self.counts.get(name, 0) + int(count)

    def maximum(self, name, value):
        self.counts[name] = max(self.counts.get(name, 0), int(value))

    def merge(self, other):
        for key, value in other.seconds.items():
            self.seconds[key] = self.seconds.get(key, 0.) + value
        for key, value in other.calls.items():
            self.calls[key] = self.calls.get(key, 0) + value
        for key, value in other.counts.items():
            if key.startswith('peak_'):
                self.maximum(key, value)
            else:
                self.count(key, value)

    def result(self, original_frames, includes_checkpoint_loading=False):
        if not isinstance(original_frames, (int, np.integer)) or original_frames < 0:
            raise ValueError('Original frame count must be a nonnegative integer')
        elapsed = time.perf_counter() - self.started if self.enabled else None
        model_seconds = sum(self.seconds.get(key, 0.) for key in ('tracknet_model', 'inpaintnet_model'))
        return {
            'enabled': self.enabled,
            'end_to_end_seconds': elapsed,
            'model_only_seconds': model_seconds if self.enabled else None,
            'original_frames': int(original_frames),
            'model_only_fps': original_frames / model_seconds if model_seconds > 0 else None,
            'end_to_end_fps': original_frames / elapsed if elapsed is not None and elapsed > 0 else None,
            'stage_seconds': dict(self.seconds),
            'stage_calls': dict(self.calls),
            'counts': dict(self.counts),
            'protocol': {
                'clock': 'time.perf_counter',
                'accelerator_synchronization': 'before and after each device stage',
                'fps_numerator': 'original output frames, never windows or padding',
                'model_only': 'TrackNet and eligible InpaintNet forward calls only',
                'warmup': 'none; first forward calls are included',
                'profiling_overhead': 'synchronization and instrumentation are included',
                'end_to_end': 'input setup, decode/preprocess, transfers, models, postprocess, metrics, and requested outputs',
                'includes_checkpoint_loading': includes_checkpoint_loading,
                'excludes': ['argument parsing', 'final run metadata assembly and JSON serialization'],
                'nested_stages_are_additive': False,
                'input_stage': 'DataLoader wait/collation plus any decode/preprocess inside it',
                'worker_timing': 'input is wait time, not summed background-worker CPU time',
            },
        }


def file_identity(path, optional=False):
    path = os.fspath(path)
    if optional and not os.path.isfile(path):
        return {'path': path, 'available': False}
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': path, 'available': True, 'bytes': os.path.getsize(path), 'sha256': digest.hexdigest()}


def checkpoint_identity(path, parameters):
    return dict(file_identity(path), parameters=parameters)


def load_checkpoint(path, record_identity=False, map_location=None, timer=None):
    """Hash the same open file used for loading, not a possibly newer checkpoint."""
    if not record_identity:
        return torch.load(path, map_location=map_location), None
    timer = timer if timer is not None else StageTimer()
    with open(path, 'rb') as source:
        before = os.fstat(source.fileno())
        checkpoint = torch.load(source, map_location=map_location)
        digest = hashlib.sha256()
        with timer.measure('checkpoint_hash'):
            source.seek(0)
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        after = os.fstat(source.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Checkpoint changed while loading or hashing: {path}')
    identity = {
        'path': os.fspath(path), 'available': True, 'bytes': after.st_size,
        'sha256': digest.hexdigest(), 'parameters': dict(checkpoint['param_dict']),
    }
    return checkpoint, identity


def model_device(model):
    parameter = next(model.parameters(), None)
    if parameter is None:
        parameter = next(model.buffers(), None)
    return parameter.device if parameter is not None else torch.device('cpu')


def runtime_environment(device):
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'opencv': cv2.__version__,
        'pillow': PIL.__version__,
        'device': str(device),
        'accelerator_name': torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else None,
        'input_dtype': 'float32',
        'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'cuda_runtime': torch.version.cuda,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
    }


def evaluation_protocol(tolerance=4., dropped_ranges=None):
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    return {
        'coordinate_space': 'TrackNet input pixels',
        'input_size': [WIDTH, HEIGHT],
        'distance': 'Euclidean distance; TP when distance <= tolerance',
        'tolerance': tolerance,
        'coordinate_quantization': 'truncate each input-pixel coordinate before distance calculation',
        'missing': 'Visibility=0 canonicalizes coordinates to (0,0)',
        'frame_alignment': 'exact Frame ID match, no padding or window duplicates',
        'classes': ['TP', 'TN', 'FP1', 'FP2', 'FN'],
        'metric_definitions': {
            'accuracy': '(TP+TN)/(TP+TN+FP1+FP2+FN)',
            'precision': 'TP/(TP+FP1+FP2)',
            'recall': 'TP/(TP+FN)',
            'f1': '2*precision*recall/(precision+recall)',
            'miss_rate': 'FN/(TP+TN+FP1+FP2+FN)',
            'zero_denominator': '0, matching the existing get_metric helper',
        },
        'confidence': 'no calibrated confidence is computed',
        'effective_trajectory_filter': dropped_ranges,
        'repair_policy': 'both raw bounding detections in one contiguous model window; no extrapolation',
    }


def prediction_statistics(pred):
    frames = pred['Frame']
    if any(len(pred[key]) != len(frames) for key in ('X', 'Y', 'Visibility')):
        raise ValueError('Prediction columns must have equal lengths')
    if len(set(frames)) != len(frames):
        raise ValueError('Output frame IDs must be unique')
    visibility = np.asarray(pred['Visibility'])
    if not np.isin(visibility, [0, 1]).all():
        raise ValueError('Visibility must contain only 0 or 1')
    longest, current = 0, 0
    for visible in visibility:
        current = 0 if visible else current + 1
        longest = max(longest, current)
    return {
        'frames': len(frames),
        'first_frame': int(frames[0]) if frames else None,
        'last_frame': int(frames[-1]) if frames else None,
        'visible_frames': int(visibility.sum()),
        'missing_frames': int(len(frames) - visibility.sum()),
        'longest_missing_run': longest,
    }


def raw_prediction(pred):
    return {
        'Frame': list(pred['Frame']),
        **{key: list(pred[f'{key}_Raw']) for key in ('X', 'Y', 'Visibility')},
    }


def video_result(raw, final, metadata, timing, raw_metrics=None, final_metrics=None, ground_truth=None):
    if raw['Frame'] != final['Frame']:
        raise ValueError('Raw and final predictions must preserve the same frame IDs')
    if (raw_metrics is None) != (final_metrics is None):
        raise ValueError('Raw and final metrics must be provided together')
    return {
        'video': metadata,
        'statistics': {
            'raw': prediction_statistics(raw),
            'final': prediction_statistics(final),
            'eligible_repair_frames': int(sum(final.get('Inpaint_Mask', []))),
            'repaired_visible_frames': sum(a == 0 and b == 1 for a, b in zip(raw['Visibility'], final['Visibility'])),
        },
        'ground_truth': ground_truth,
        'metrics': None if raw_metrics is None else {'raw': raw_metrics, 'final': final_metrics},
        'evaluated_frames': None if raw_metrics is None else sum(raw_metrics[key] for key in PRED_TYPES),
        'timing': timing,
    }


def write_run_report(path, settings, models, videos, timing, device, protocol=None):
    metrics = None
    if videos and all(video['metrics'] is not None for video in videos):
        metrics = {
            stage: metrics_from_counts({
                key: sum(video['metrics'][stage][key] for video in videos) for key in PRED_TYPES
            }) for stage in ('raw', 'final')
        }
    root = os.path.dirname(os.path.dirname(__file__))
    modules = [('model.py',), ('predict.py',), ('test.py',), ('dataset.py',), ('generate_mask_data.py',),
               ('utils', 'general.py'),
               ('utils', 'inpaint.py'), ('utils', 'temporal.py'), ('utils', 'runtime.py'), ('utils', 'metric.py')]
    report = {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'settings': settings,
        'models': models,
        'implementation_sha256': {
            '.'.join(parts): file_identity(os.path.join(root, *parts))['sha256'] for parts in modules
        },
        'runtime': runtime_environment(device),
        'evaluation_protocol': protocol,
        'videos': videos,
        'metrics': metrics,
        'timing': timing,
    }
    with open(path, 'w', encoding='utf8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    return report
