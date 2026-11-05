import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader

import dataset as datasets
import test as evaluation
with contextlib.redirect_stdout(io.StringIO()):
    import predict as inference
from test_inpainting import ConstantInpaint
from utils.general import HEIGHT, WIDTH
from utils.inpaint import get_ensemble_weight
from utils.metric import metrics_from_types, validate_heatmap_threshold
from utils.runtime import StageTimer, checkpoint_identity, evaluation_protocol, file_identity, load_checkpoint, video_result, write_run_report
from utils.temporal import iter_heatmap_predictions


ROOT = Path(__file__).resolve().parents[1]
MODES = ('nonoverlap', 'average', 'weight')


def load_json(path):
    with open(path, encoding='utf8') as source:
        return json.load(source)


class CaptureFactory:
    def __init__(self, frames, fps=4., shape=(9, 13)):
        self.frames, self.fps, self.shape = frames, fps, shape
        self.captures = []

    def __call__(self, path):
        factory = self

        class Capture:
            def __init__(self):
                self.position, self.closed, self.samples = 0, False, []

            def isOpened(self):
                return True

            def get(self, prop):
                h, w = factory.frames[0].shape[:2] if factory.frames else factory.shape
                return {cv2.CAP_PROP_FRAME_COUNT: len(factory.frames), cv2.CAP_PROP_FPS: factory.fps,
                        cv2.CAP_PROP_FRAME_WIDTH: w, cv2.CAP_PROP_FRAME_HEIGHT: h}.get(prop, 0)

            def set(self, prop, value):
                self.position = int(value)
                return True

            def read(self):
                if self.position >= len(factory.frames):
                    return False, None
                self.samples.append(self.position)
                frame = factory.frames[self.position].copy()
                self.position += 1
                return True, frame

            def release(self):
                self.closed = True

        capture = Capture()
        self.captures.append(capture)
        return capture


class StreamingTests(unittest.TestCase):
    def test_preprocessing_matches_existing_dataset_and_caches_each_frame_once(self):
        rng = np.random.default_rng(42)
        for mode in ('', 'subtract', 'subtract_concat', 'concat'):
            for step in (1, 3):
                for length in (0, 1, 2, 3, 6, 7):
                    with self.subTest(mode=mode, step=step, length=length):
                        frames = [rng.integers(0, 256, (9, 13, 3), dtype=np.uint8) for _ in range(length)]
                        rgb = np.asarray(frames)[..., ::-1] if frames else None
                        median = np.median(rgb, axis=0) if frames else np.zeros((9, 13, 3))
                        if frames:
                            oracle_step = 3 if length < 3 else step
                            oracle = datasets.Shuttlecock_Trajectory_Dataset(
                                frame_arr=rgb, seq_len=3, sliding_step=oracle_step,
                                padding=oracle_step == 3, bg_mode=mode, median=median, HEIGHT=6, WIDTH=8)
                            expected = [oracle[i] for i in range(len(oracle))]
                            provided = oracle.median if mode == 'concat' else median
                        else:
                            expected = []
                            provided = np.zeros((3, 6, 8), dtype=np.uint8) if mode == 'concat' else median
                        factory = CaptureFactory(frames)
                        with patch('dataset.cv2.VideoCapture', factory):
                            stream = datasets.Video_IterableDataset(
                                'synthetic.mp4', seq_len=3, sliding_step=step, bg_mode=mode,
                                median=provided, HEIGHT=6, WIDTH=8)
                            with patch.object(stream, '_process_frame', wraps=stream._process_frame) as process:
                                actual = list(stream)
                            self.assertEqual(process.call_count, length)
                        self.assertEqual(len(actual), len(expected))
                        for (indices, values), (reference_indices, reference) in zip(actual, expected):
                            np.testing.assert_array_equal(indices, reference_indices)
                            self.assertEqual(values.dtype, np.float32)
                            np.testing.assert_allclose(values, reference.astype(np.float32), rtol=0, atol=1e-7)
                        self.assertEqual(stream.stats['decoded_frames'], length)
                        self.assertEqual(stream.stats['preprocessed_frames'], length)
                        self.assertLessEqual(stream.stats['peak_cached_frames'], 3)
                        channels = 1 if mode == 'subtract' else 4 if mode == 'subtract_concat' else 3
                        self.assertLessEqual(stream.stats['peak_cache_bytes'], 3 * channels * 6 * 8 * 4)
                        self.assertTrue(all(capture.closed for capture in factory.captures))

    def test_iteration_can_restart_and_early_close_releases_decoder(self):
        frames = [np.full((9, 13, 3), i, dtype=np.uint8) for i in range(7)]
        factory = CaptureFactory(frames)
        with patch('dataset.cv2.VideoCapture', factory):
            stream = datasets.Video_IterableDataset('synthetic.mp4', seq_len=3, HEIGHT=6, WIDTH=8)
            first, second = list(stream), list(stream)
            self.assertEqual([i.tolist() for i, _ in first], [i.tolist() for i, _ in second])
            iterator = iter(stream)
            next(iterator)
            iterator.close()
        self.assertTrue(all(capture.closed for capture in factory.captures))

    def test_background_limit_ranges_and_exact_original_resolution_median(self):
        frames = [np.full((9, 13, 3), i, dtype=np.uint8) for i in range(20)]
        for limit, seconds in ((100, None), (10, None), (7, None), (5, [1, 4])):
            with self.subTest(limit=limit, seconds=seconds):
                factory = CaptureFactory(frames)
                with patch('dataset.cv2.VideoCapture', factory), contextlib.redirect_stdout(io.StringIO()):
                    stream = datasets.Video_IterableDataset(
                        'synthetic.mp4', bg_mode='subtract', max_sample_num=limit,
                        video_range=seconds, HEIGHT=6, WIDTH=8)
                start, end = (0, 20) if seconds is None else (4, 16)
                indices = list(range(start, end, max(1, math.ceil((end - start) / limit))))
                actual = stream.stats['background']
                self.assertEqual(actual['sample_indices'], indices)
                self.assertLessEqual(len(indices), limit)
                self.assertEqual(factory.captures[1].samples, indices)
                expected = np.median(np.asarray(frames)[indices], axis=0)[..., ::-1]
                np.testing.assert_array_equal(stream.median, expected)
                self.assertEqual(stream.median.shape, (9, 13, 3))
                self.assertEqual(stream.median.dtype, np.float64)
                self.assertEqual(actual['sample_buffer_bytes'], len(indices) * 9 * 13 * 3)
                self.assertTrue(all(capture.closed for capture in factory.captures))

    def test_fractional_median_empty_video_and_invalid_sampling_settings(self):
        factory = CaptureFactory([np.zeros((9, 13, 3), np.uint8), np.ones((9, 13, 3), np.uint8)])
        with patch('dataset.cv2.VideoCapture', factory), contextlib.redirect_stdout(io.StringIO()):
            stream = datasets.Video_IterableDataset('synthetic.mp4', bg_mode='subtract')
            self.assertTrue(np.all(stream.median == 0.5))
        with patch('dataset.cv2.VideoCapture', CaptureFactory([])), contextlib.redirect_stdout(io.StringIO()):
            empty = datasets.Video_IterableDataset('empty.mp4', bg_mode='concat')
            self.assertEqual(list(empty), [])
            self.assertEqual(empty.stats['background']['method'], 'empty_video')
        for limit in (0, -1, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                datasets.Video_IterableDataset('synthetic.mp4', max_sample_num=limit)
        for seconds in ([2, 1], [-1, 1], [1, 1], [1], [float('nan'), 2], [9, 10]):
            with self.subTest(seconds=seconds), patch('dataset.cv2.VideoCapture', factory):
                with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                    datasets.Video_IterableDataset('synthetic.mp4', bg_mode='subtract', video_range=seconds)

    def test_fractional_fps_samples_stay_inside_the_requested_time_range(self):
        frames = [np.full((9, 13, 3), i, np.uint8) for i in range(20)]
        factory = CaptureFactory(frames, fps=2.5)
        with patch('dataset.cv2.VideoCapture', factory), contextlib.redirect_stdout(io.StringIO()):
            stream = datasets.Video_IterableDataset(
                'synthetic.mp4', bg_mode='subtract', video_range=[1, 3], max_sample_num=3)
        indices = stream.stats['background']['sample_indices']
        self.assertEqual(indices, [3, 5, 7])
        self.assertTrue(all(1 <= index / 2.5 < 3 for index in indices))


class TemporalTests(unittest.TestCase):
    @staticmethod
    def old_ensemble(predictions, seq_len, mode):
        buffer = torch.cat((torch.zeros((seq_len - 1, seq_len, 2, 3)), predictions,
                            torch.zeros((seq_len - 1, seq_len, 2, 3))))
        batch_i, frame_i = torch.arange(seq_len), torch.arange(seq_len - 1, -1, -1)
        result = []
        for index in range(len(predictions)):
            diagonal = buffer[batch_i + index, frame_i]
            result.append(diagonal.sum(0) / (index + 1) if index < seq_len - 1
                          else (diagonal * get_ensemble_weight(seq_len, mode)[:, None, None]).sum(0))
        for offset in range(1, seq_len):
            result.append(buffer[batch_i + len(predictions) - 1 + offset, frame_i].sum(0) / (seq_len - offset))
        return torch.stack(result)

    def test_all_modes_match_previous_ensemble_and_preserve_real_frames(self):
        generator = torch.Generator().manual_seed(7)
        for length in (0, 1, 2, 3, 4, 5, 6, 7, 8):
            for seq_len in (1, 3, 4):
                for mode in MODES:
                    for batch_size in (1, 3):
                        with self.subTest(length=length, seq_len=seq_len, mode=mode, batch_size=batch_size):
                            starts = (list(range(0, length, seq_len)) if mode == 'nonoverlap'
                                      else list(range(max(1, length - seq_len + 1))) if length else [])
                            positions = torch.tensor([[min(start + f, length - 1) for f in range(seq_len)]
                                                      for start in starts], dtype=torch.long).reshape(-1, seq_len)
                            indices = torch.stack((torch.zeros_like(positions), positions), dim=-1)
                            heatmaps = torch.rand((len(starts), seq_len, 2, 3), generator=generator)
                            batches = [(indices[i:i + batch_size], heatmaps[i:i + batch_size])
                                       for i in range(0, len(starts), batch_size)]
                            timer = StageTimer()
                            output = list(iter_heatmap_predictions(torch.nn.Identity(), batches, mode, 'cpu', timer))
                            emitted = [int(index) for ids, _, _ in output for index in ids[:, 0, 1]]
                            self.assertEqual(emitted, list(range(length)))
                            self.assertEqual(timer.counts.get('tracknet_windows', 0), len(starts))
                            self.assertEqual(timer.counts.get('tracknet_model_frame_evaluations', 0), len(starts) * seq_len)
                            self.assertLessEqual(timer.counts.get('peak_pending_frames', 0), seq_len)
                            if length:
                                actual = torch.cat([values[:, 0] for _, _, values in output])
                                if mode == 'nonoverlap' or length < seq_len:
                                    expected = torch.stack([heatmaps[n, f] for n in range(len(starts))
                                                            for f in range(seq_len)
                                                            if f == 0 or positions[n, f] != positions[n, f - 1]])
                                elif length >= 2 * seq_len - 2:
                                    expected = self.old_ensemble(heatmaps, seq_len, mode)
                                else:
                                    expected = torch.stack([
                                        torch.stack([heatmaps[n, f] for n in range(len(starts))
                                                     for f in range(seq_len) if int(positions[n, f]) == frame]).mean(0)
                                        for frame in range(length)
                                    ])
                                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_short_overlap_never_divides_by_nonexistent_windows(self):
        for seq_len in (3, 4):
            for length in range(seq_len, 2 * seq_len - 2):
                for mode in ('average', 'weight'):
                    with self.subTest(seq_len=seq_len, length=length, mode=mode):
                        positions = torch.tensor([[start + f for f in range(seq_len)]
                                                  for start in range(length - seq_len + 1)])
                        indices = torch.stack((torch.zeros_like(positions), positions), dim=-1)
                        heatmaps = torch.full((len(positions), seq_len, 2, 3), 0.6)
                        output = list(iter_heatmap_predictions(
                            torch.nn.Identity(), [(indices, heatmaps)], mode, 'cpu'))
                        actual = torch.cat([prediction[:, 0] for _, _, prediction in output])
                        torch.testing.assert_close(actual, torch.full((length, 2, 3), 0.6))


class ThresholdAndReportTests(unittest.TestCase):
    def test_threshold_default_strict_comparison_bounds_and_coordinate_independence(self):
        indices = torch.tensor([[[0, 0]]])
        truth = torch.zeros((1, 1, 3, 3))
        truth[0, 0, 1, 1] = 1.
        prediction = truth * 0.5
        self.assertEqual(inference.predict(indices, y_pred=prediction)['Visibility'], [0])
        self.assertEqual(inference.predict(indices, y_pred=prediction, heatmap_threshold=0.49)['Visibility'], [1])
        self.assertEqual(evaluation.evaluate(indices, y_true=truth, y_pred=prediction)['Type'], [4])
        self.assertEqual(evaluation.evaluate(indices, y_true=truth, y_pred=prediction, heatmap_threshold=0.49)['Type'], [0])
        self.assertEqual(inference.predict(indices, c_pred=torch.full((1, 1, 2), 0.5),
                                           heatmap_threshold=1.)['Visibility'], [1])
        for value in (0., 0.5, 1.):
            self.assertEqual(validate_heatmap_threshold(value), value)
        for value in (-0.1, 1.1, float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_heatmap_threshold(value)
        for invalid in ([5], [float('nan')], [[0]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                metrics_from_types(invalid)
        self.assertEqual(evaluation.get_test_res({}, drop=True)['TP'], 0)
        self.assertEqual(evaluation.get_coco_res({}, drop=True), [])

    def test_training_evaluation_uses_the_effective_heatmap_threshold(self):
        indices = torch.tensor([[[0, 0]]])
        truth = torch.zeros((1, 1, 3, 3))
        truth[0, 0, 1, 1] = 1.
        loader = [(indices, torch.full_like(truth, 0.5), truth, None, None)]
        for threshold, expected in ((0.5, 'FN'), (0.49, 'TP')):
            with self.subTest(threshold=threshold), patch.object(torch.Tensor, 'cuda', lambda value: value):
                _, result = evaluation.eval_tracknet(torch.nn.Identity(), loader,
                                                     {'verbose': False, 'tolerance': 4.,
                                                      'heatmap_threshold': threshold})
                self.assertEqual(result[expected], 1)

    def test_prediction_metrics_align_frames_and_do_not_reward_invented_repairs(self):
        raw = {'Frame': [100, 101, 102], 'X': [100, 0, 200], 'Y': [100, 0, 100], 'Visibility': [1, 0, 1]}
        final = dict(raw, X=[100, 150, 200], Y=[100, 100, 100], Visibility=[1, 1, 1], Inpaint_Mask=[0, 1, 0])
        labels = pd.DataFrame(raw).iloc[::-1]
        raw_metrics = metrics_from_types(evaluation.evaluate_prediction_dict(raw, labels, (WIDTH, HEIGHT))['Type'])
        final_metrics = metrics_from_types(evaluation.evaluate_prediction_dict(final, labels, (WIDTH, HEIGHT))['Type'])
        self.assertEqual(raw_metrics['precision'], 1.)
        self.assertAlmostEqual(final_metrics['precision'], 2 / 3)
        self.assertEqual(final_metrics['FP2'], 1)
        result = video_result(raw, final, {}, None, raw_metrics, final_metrics, {'source': 'synthetic labels'})
        self.assertEqual(result['statistics']['repaired_visible_frames'], 1)
        labels = pd.DataFrame({'Frame': [100], 'X': [100], 'Y': [100], 'Visibility': [1]})
        prediction = {'Frame': [100], 'X': [108], 'Y': [100], 'Visibility': [1]}
        self.assertEqual(evaluation.evaluate_prediction_dict(prediction, labels, (1024, 576), 4.)['Type'], [0])
        prediction['X'] = [110]
        self.assertEqual(evaluation.evaluate_prediction_dict(prediction, labels, (1024, 576), 4.)['Type'], [2])
        with self.assertRaises(ValueError):
            evaluation.evaluate_prediction_dict(raw, labels, (WIDTH, HEIGHT))

    def test_timer_counts_original_frames_and_synchronizes_only_when_enabled(self):
        with patch('utils.runtime.time.perf_counter', side_effect=[10., 12., 15., 20.]):
            with patch.object(StageTimer, 'synchronize') as synchronize:
                timer = StageTimer(True)
                with timer.measure('tracknet_model', 'cuda:0'):
                    pass
                timer.count('tracknet_windows', 7)
                timer.count('tracknet_model_frame_evaluations', 28)
                result = timer.result(5)
        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual(result['model_only_seconds'], 3.)
        self.assertEqual(result['end_to_end_seconds'], 10.)
        self.assertAlmostEqual(result['model_only_fps'], 5 / 3)
        self.assertEqual(result['end_to_end_fps'], 0.5)
        self.assertEqual(result['original_frames'], 5)
        with patch('utils.runtime.time.perf_counter', side_effect=AssertionError('disabled timer read the clock')):
            with patch.object(StageTimer, 'synchronize') as synchronize:
                timer = StageTimer()
                with timer.measure('tracknet_model', 'cuda'):
                    pass
                self.assertIsNone(timer.result(5)['model_only_fps'])
                synchronize.assert_not_called()
        for invalid in (-1, 0.5, float('nan')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                StageTimer().result(invalid)
        with patch('torch.cuda.synchronize') as cuda_sync:
            StageTimer.synchronize('cuda:0')
            cuda_sync.assert_called_once_with(torch.device('cuda:0'))
        with patch('torch.mps.synchronize') as mps_sync:
            StageTimer.synchronize('mps')
            mps_sync.assert_called_once()

    def test_report_contains_identities_protocol_and_no_unlabeled_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pt'
            checkpoint.write_bytes(b'synthetic checkpoint identity')
            identity = checkpoint_identity(checkpoint, {'seq_len': 4})
            self.assertEqual(identity['sha256'], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            raw = {'Frame': [0], 'X': [0], 'Y': [0], 'Visibility': [0]}
            result = video_result(raw, dict(raw, Inpaint_Mask=[0]), {'width': 32, 'height': 24},
                                  StageTimer().result(1))
            report_file = Path(directory) / 'run.json'
            write_run_report(report_file, {'heatmap_threshold': 0.5}, {'TrackNet': identity},
                             [result], StageTimer().result(1), 'cpu', evaluation_protocol())
            report = load_json(report_file)
            self.assertEqual(report['schema_version'], 1)
            self.assertIsNone(report['metrics'])
            self.assertIsNone(report['videos'][0]['metrics'])
            self.assertNotIn('precision', report['videos'][0]['statistics'])
            self.assertEqual(report['evaluation_protocol']['input_size'], [WIDTH, HEIGHT])
            self.assertTrue(report['implementation_sha256'])
            self.assertFalse(file_identity(Path(directory) / 'absent.mp4', optional=True)['available'])

    def test_checkpoint_identity_is_frozen_at_load_and_detects_concurrent_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            path.write_bytes(b'original synthetic checkpoint')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            payload = {'param_dict': {'seq_len': 4}, 'model': {}}
            with patch('torch.load', return_value=payload):
                _, identity = load_checkpoint(path, record_identity=True)
            path.write_bytes(b'a later checkpoint at the same path')
            self.assertEqual(identity['sha256'], expected)

            def changing_checkpoint(source, **kwargs):
                status = path.stat()
                os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 10_000_000))
                return payload

            with patch('torch.load', side_effect=changing_checkpoint), self.assertRaisesRegex(ValueError, 'changed while'):
                load_checkpoint(path, record_identity=True)


class VideoModel(torch.nn.Module):
    def __init__(self, bg_mode=''):
        super().__init__()
        self.bg_mode = bg_mode

    def forward(self, frames):
        if self.bg_mode == 'concat':
            frames = frames[:, 3:]
        frame_ids = ((frames[:, ::3, 0, 0] * 255 - 30) / 20).round().long()
        heatmaps = torch.zeros((len(frames), frame_ids.shape[1], HEIGHT, WIDTH), dtype=frames.dtype)
        for n in range(len(frames)):
            for f, frame_id in enumerate(frame_ids[n].tolist()):
                if frame_id in (0, 1, 3, 7):
                    heatmaps[n, f, HEIGHT // 2, WIDTH // 2] = 0.6
        return heatmaps


class PipelineTests(unittest.TestCase):
    def run_cli(self, script, arguments, data_root=None, bg_mode=''):
        tracker, inpainter = VideoModel(bg_mode), ConstantInpaint()

        def checkpoint(path, **kwargs):
            path = getattr(path, 'name', path)
            if Path(path).name == 'tracknet.pt':
                return {'param_dict': {'model_name': 'TrackNet', 'seq_len': 3, 'bg_mode': bg_mode, 'tolerance': 4.},
                        'model': tracker.state_dict()}
            return {'param_dict': {'seq_len': 4}, 'model': inpainter.state_dict()}

        def loader(dataset, **kwargs):
            kwargs['num_workers'] = 0
            return DataLoader(dataset, **kwargs)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', [script] + arguments))
            stack.enter_context(patch('torch.load', side_effect=checkpoint))
            stack.enter_context(patch('torch.cuda.is_available', return_value=False))
            stack.enter_context(patch.object(torch.backends, 'mps', types.SimpleNamespace(is_available=lambda: False), create=True))
            stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda value: value))
            stack.enter_context(patch('torch.utils.data.DataLoader', side_effect=loader))
            stack.enter_context(patch('tqdm.tqdm', side_effect=lambda values: values))
            stack.enter_context(patch('dataset.tqdm', side_effect=lambda values: values))
            stack.enter_context(patch('utils.general.get_model',
                                      side_effect=lambda name, *args, **kwargs: tracker if name == 'TrackNet' else inpainter))
            if data_root is not None:
                stack.enter_context(patch.object(datasets, 'data_dir', str(data_root)))
                stack.enter_context(patch.object(evaluation, 'data_dir', str(data_root)))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return runpy.run_path(str(ROOT / script), run_name='__main__')

    def make_inputs(self, directory):
        root = Path(directory)
        for name in ('tracknet.pt', 'inpaintnet.pt'):
            (root / name).write_bytes(b'synthetic model stub, not trained weights')
        frames = [np.full((24, 32, 3), 30 + i * 20, dtype=np.uint8) for i in range(9)]
        labels = pd.DataFrame({
            'Frame': list(range(9)), 'Visibility': [1, 1, 0, 1, 0, 0, 0, 1, 0],
            'X': [16, 16, 0, 16, 0, 0, 0, 16, 0], 'Y': [12, 12, 0, 12, 0, 0, 0, 12, 0],
        })
        return root, frames, labels

    def make_clip(self, path, frames):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10., (32, 24))
        if not writer.isOpened():
            self.skipTest('OpenCV mp4v writer unavailable')
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

    def test_invalid_cli_thresholds_are_rejected_before_loading_weights(self):
        for script in ('predict.py', 'test.py', 'generate_mask_data.py'):
            with self.subTest(script=script), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.run_cli(script, ['--heatmap_threshold', 'nan'])
                self.assertEqual(error.exception.code, 2)

    def test_empty_and_short_video_outputs_preserve_frames_in_both_paths(self):
        for length in (0, 1, 2):
            for mode in MODES:
                for streaming in (False, True):
                    with self.subTest(length=length, mode=mode, streaming=streaming):
                        frames = [np.full((24, 32, 3), 30 + i * 20, np.uint8) for i in range(length)]
                        factory = CaptureFactory(frames, shape=(24, 32))
                        settings = {'tracknet_seq_len': 3, 'inpaintnet_seq_len': 4, 'bg_mode': '',
                                    'eval_mode': mode, 'large_video': streaming, 'batch_size': 2,
                                    'num_workers': 0, 'max_sample_num': 5, 'video_range': None,
                                    'heatmap_threshold': 0.5}
                        timer = StageTimer(True)
                        with patch('dataset.cv2.VideoCapture', factory), contextlib.redirect_stderr(io.StringIO()):
                            raw, final, metadata = inference.predict_video(
                                VideoModel(), 'synthetic.mp4', settings, ConstantInpaint(), timer, 'cpu')
                        self.assertEqual(final['Frame'], list(range(length)))
                        self.assertEqual(final['Visibility'], [1] * length)
                        self.assertEqual(raw['Frame'], final['Frame'])
                        self.assertEqual(metadata['processed_frames'], length)
                        self.assertEqual(timer.result(length)['original_frames'], length)
                        self.assertEqual(timer.counts.get('tracknet_windows', 0), int(length > 0))

    def test_real_synthetic_clip_modes_thresholds_sidecars_and_csv_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root, frames, labels = self.make_inputs(directory)
            clip = root / 'sample.mp4'
            self.make_clip(clip, frames)
            label_file = root / 'labels.csv'
            labels.to_csv(label_file, index=False)
            for mode in MODES:
                baseline = None
                for streaming in (False, True):
                    output = root / f'{mode}_{streaming}'
                    args = ['--video_file', str(clip), '--tracknet_file', str(root / 'tracknet.pt'),
                            '--inpaintnet_file', str(root / 'inpaintnet.pt'), '--batch_size', '2',
                            '--save_dir', str(output), '--eval_mode', mode, '--output_run', '--label_file', str(label_file)]
                    if streaming:
                        args.append('--large_video')
                    self.run_cli('predict.py', args)
                    prediction = pd.read_csv(output / 'sample_ball.csv')
                    self.assertEqual(prediction.columns.tolist(), ['Frame', 'Visibility', 'X', 'Y'])
                    self.assertEqual(prediction['Frame'].tolist(), list(range(9)))
                    self.assertEqual(prediction['Visibility'].tolist(), [1, 1, 1, 1, 0, 0, 0, 1, 0])
                    if baseline is None:
                        baseline = prediction
                    else:
                        pd.testing.assert_frame_equal(prediction, baseline)
                    report = load_json(output / 'sample_run.json')
                    self.assertEqual(report['metrics']['raw']['FP2'], 0)
                    self.assertEqual(report['metrics']['final']['FP2'], 1)
                    self.assertLess(report['metrics']['final']['precision'], report['metrics']['raw']['precision'])
                    self.assertEqual(report['timing']['original_frames'], 9)
                    self.assertEqual(report['settings']['heatmap_threshold'], 0.5)
                    self.assertEqual(report['videos'][0]['video']['fps'], 10.)
                    counts = report['timing']['counts']
                    self.assertEqual(counts['tracknet_windows'], 3 if mode == 'nonoverlap' else 7)
                    if streaming:
                        self.assertEqual(counts['preprocessed_frames'], 9)
                        self.assertLessEqual(counts['peak_cached_frames'], 3)
            output = root / 'threshold'
            self.run_cli('predict.py', ['--video_file', str(clip), '--tracknet_file', str(root / 'tracknet.pt'),
                                       '--inpaintnet_file', str(root / 'inpaintnet.pt'), '--large_video',
                                       '--save_dir', str(output), '--output_run', '--heatmap_threshold', '0.7'])
            self.assertEqual(pd.read_csv(output / 'sample_ball.csv')['Visibility'].sum(), 0)
            report = load_json(output / 'sample_run.json')
            self.assertIsNone(report['metrics'])
            self.assertEqual(report['settings']['heatmap_threshold'], 0.7)
            self.assertEqual(report['timing']['counts'].get('inpaintnet_windows', 0), 0)
            output = root / 'no_sidecar'
            self.run_cli('predict.py', ['--video_file', str(clip), '--tracknet_file', str(root / 'tracknet.pt'),
                                       '--large_video', '--save_dir', str(output)])
            self.assertEqual(pd.read_csv(output / 'sample_ball.csv').columns.tolist(), ['Frame', 'Visibility', 'X', 'Y'])
            self.assertFalse((output / 'sample_run.json').exists())

    def make_rally(self, directory):
        root, frames, labels = self.make_inputs(directory)
        data_root = root / 'data'
        match = data_root / 'test' / 'match1'
        rally = match / 'frame' / 'rally'
        rally.mkdir(parents=True)
        (match / 'corrected_csv').mkdir()
        for i, frame in enumerate(frames):
            Image.fromarray(frame[..., ::-1]).save(rally / f'{i}.png')
        labels.to_csv(match / 'corrected_csv' / 'rally_ball.csv', index=False)
        with open(data_root / 'drop_frame.json', 'w') as output:
            json.dump({'start': {'1_rally': 1}, 'end': {'1_rally': 8}}, output)
        return root, data_root, match, rally

    def test_concat_background_defaults_match_preloaded_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root, frames, _ = self.make_inputs(directory)
            clip = root / 'sample.mp4'
            self.make_clip(clip, frames)
            for mode in MODES:
                predictions = []
                for streaming in (False, True):
                    output = root / f'concat_{mode}_{streaming}'
                    arguments = ['--video_file', str(clip), '--tracknet_file', str(root / 'tracknet.pt'),
                                 '--inpaintnet_file', str(root / 'inpaintnet.pt'), '--batch_size', '2',
                                 '--eval_mode', mode, '--save_dir', str(output), '--output_run']
                    if streaming:
                        arguments.append('--large_video')
                    self.run_cli('predict.py', arguments, bg_mode='concat')
                    predictions.append(pd.read_csv(output / 'sample_ball.csv'))
                    report = load_json(output / 'sample_run.json')
                    if streaming:
                        background = report['videos'][0]['video']['streaming']['background']
                        self.assertEqual(background['sample_indices'], list(range(9)))
                        self.assertEqual(report['timing']['counts']['preprocessed_frames'], 9)
                pd.testing.assert_frame_equal(predictions[0], predictions[1])

    def test_labeled_evaluation_uses_supplied_tracker_and_records_raw_final_filtered_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root, data_root, match, _ = self.make_rally(directory)
            output = root / 'evaluation'
            self.assertFalse((match / 'predicted_csv').exists())
            self.run_cli('test.py', [
                '--tracknet_file', str(root / 'tracknet.pt'), '--inpaintnet_file', str(root / 'inpaintnet.pt'),
                '--output_run', '--save_dir', str(output), '--batch_size', '2',
            ], data_root)
            report = load_json(output / 'test_run_weight.json')
            result = report['videos'][0]
            self.assertEqual(result['video']['raw_source'], 'fresh_tracknet')
            self.assertEqual(result['evaluated_frames'], 7)
            self.assertEqual(result['timing']['original_frames'], 9)
            self.assertEqual(result['metrics']['raw']['FP2'], 0)
            self.assertEqual(result['metrics']['raw']['TP'], 3)
            self.assertEqual(result['metrics']['final']['FP2'], 1)
            self.assertEqual(report['metrics']['final']['FP2'], 1)
            self.assertTrue(report['evaluation_protocol']['effective_trajectory_filter']['sha256'])

    def test_video_evaluation_and_linear_comparison_write_compatible_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root, data_root, match, _ = self.make_rally(directory)
            (match / 'video').mkdir()
            clip = match / 'video' / 'rally.mp4'
            frames = [np.full((24, 32, 3), 30 + i * 20, np.uint8) for i in range(9)]
            self.make_clip(clip, frames)
            for linear in (False, True):
                output = root / f'video_{linear}'
                arguments = ['--tracknet_file', str(root / 'tracknet.pt'),
                             '--inpaintnet_file', str(root / 'inpaintnet.pt'), '--video_file', str(clip),
                             '--output_run', '--output_provenance', '--save_dir', str(output),
                             '--eval_mode', 'nonoverlap', '--batch_size', '2']
                if linear:
                    arguments.append('--linear_interp')
                self.run_cli('test.py', arguments, data_root)
                prediction = pd.read_csv(output / 'rally_ball.csv')
                self.assertEqual(prediction['Frame'].tolist(), list(range(9)))
                self.assertEqual(prediction.columns.tolist(),
                                 ['Frame', 'Visibility', 'X', 'Y', 'Visibility_Raw', 'X_Raw', 'Y_Raw', 'Inpaint_Mask'])
                report = load_json(output / 'rally_run.json')
                self.assertEqual(report['videos'][0]['metrics']['final']['FP2'], 1)
                self.assertEqual(report['videos'][0]['video']['fps'], 10.)
                self.assertEqual(report['timing']['original_frames'], 9)
                self.assertIn('output_video', report['timing']['stage_seconds'])
                if linear:
                    self.assertEqual(report['timing']['counts'].get('inpaintnet_windows', 0), 0)
                cap = cv2.VideoCapture(str(output / 'rally.mp4'))
                try:
                    self.assertTrue(cap.isOpened())
                    self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 9)
                finally:
                    cap.release()

    def test_mask_generation_threshold_cache_identity_and_unknown_legacy_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root, data_root, match, rally = self.make_rally(directory)
            self.run_cli('generate_mask_data.py', [
                '--tracknet_file', str(root / 'tracknet.pt'), '--split_list', 'test',
                '--output_run', '--batch_size', '2', '--eval_mode', 'average',
            ], data_root)
            csv_file = match / 'predicted_csv' / 'rally_ball.csv'
            generated = pd.read_csv(csv_file)
            self.assertEqual(generated.columns.tolist(),
                             ['Frame', 'Visibility_GT', 'X_GT', 'Y_GT', 'Visibility', 'X', 'Y', 'Inpaint_Mask'])
            sidecar = Path(str(csv_file) + '.run.json')
            self.assertEqual(load_json(sidecar)['videos'][0]['prediction_csv']['sha256'], file_identity(csv_file)['sha256'])
            settings = {'inpaintnet_seq_len': 4, 'eval_mode': 'average', 'batch_size': 2,
                        'tolerance': 4., 'heatmap_threshold': 0.8}
            info = {}
            evaluation.test_rally((None, ConstantInpaint()), str(rally), settings, run_info=info)
            self.assertEqual(info['heatmap_threshold_applied'], 0.5)
            self.assertEqual(info['requested_heatmap_threshold'], 0.8)
            with open(csv_file, 'a') as output:
                output.write('\n')
            with self.assertRaisesRegex(ValueError, 'no longer matches'):
                evaluation.test_rally((None, ConstantInpaint()), str(rally), settings, run_info={})
            sidecar.unlink()
            info = {}
            evaluation.test_rally((None, ConstantInpaint()), str(rally), settings, run_info=info)
            self.assertIsNone(info['heatmap_threshold_applied'])
            self.assertIsNone(info['cache_generation'])
            self.run_cli('generate_mask_data.py', [
                '--tracknet_file', str(root / 'tracknet.pt'), '--split_list', 'test',
                '--output_run', '--batch_size', '2', '--heatmap_threshold', '0.7',
            ], data_root)
            self.assertEqual(pd.read_csv(csv_file)['Visibility'].sum(), 0)
            report = load_json(data_root / 'mask_generation_run_weight.json')
            self.assertEqual(report['settings']['heatmap_threshold'], 0.7)


if __name__ == '__main__':
    unittest.main()
