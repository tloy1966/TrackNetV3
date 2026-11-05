import contextlib
import copy
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader

import test as evaluation
from utils.general import COOR_TH, HEIGHT, WIDTH, write_pred_csv
from utils.inpaint import (
    format_inpaint_predictions, generate_inpaint_mask, get_ensemble_weight,
    inpaint_batch, inpaint_trajectory,
)


MODES = ('nonoverlap', 'average', 'weight')
POINTS = [(1, 2), (60, 80), (0, 0), (120, 100),
          (0, 0), (0, 0), (0, 0), (200, 120), (0, 0)]


class ConstantInpaint(torch.nn.Module):
    def __init__(self, value=0.5):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.value = value
        self.windows_seen = 0

    def forward(self, coordinates, mask):
        self.windows_seen += len(coordinates)
        return torch.full_like(coordinates, self.value)


class PositionInpaint(ConstantInpaint):
    def forward(self, coordinates, mask):
        positions = torch.arange(1, coordinates.shape[1] + 1, dtype=coordinates.dtype,
                                 device=coordinates.device)
        return ((positions / 6) ** 2)[None, :, None].expand_as(coordinates).clone()


class AlternatingInpaint(ConstantInpaint):
    def forward(self, coordinates, mask):
        result = torch.zeros_like(coordinates)
        result[:, 1::2, 0] = 0.1
        result[:, ::2, 1] = 0.1
        return result


class TrackingStub(torch.nn.Module):
    def forward(self, frames):
        frame_ids = (frames[:, ::3, 0, 0] * 255).round().long()
        heatmaps = torch.zeros((len(frames), frame_ids.shape[1], HEIGHT, WIDTH),
                               device=frames.device)
        for n in range(len(frames)):
            for f, frame_id in enumerate(frame_ids[n].tolist()):
                x, y = POINTS[frame_id]
                if x != 0 or y != 0:
                    heatmaps[n, f, y, x] = 1.
        return heatmaps


def trajectory(visibility):
    coordinates = torch.zeros((len(visibility), 2), dtype=torch.float32)
    observed = torch.tensor(visibility, dtype=torch.bool)
    coordinates[observed] = 0.25
    return coordinates, (~observed)[:, None].float()


class RepairPolicyTests(unittest.TestCase):
    def test_candidate_mask_requires_both_endpoints_and_context(self):
        cases = [
            ([], 4, []),
            ([0, 0, 0], 4, [0, 0, 0]),
            ([0, 1, 0, 1, 0], 4, [0, 0, 1, 0, 0]),
            ([1, 0, 0, 1], 4, [0, 1, 1, 0]),
            ([1, 0, 0, 0, 1], 4, [0, 0, 0, 0, 0]),
            ([1, 0, 1], 2, [0, 0, 0]),
        ]
        for visibility, seq_len, expected in cases:
            with self.subTest(visibility=visibility, seq_len=seq_len):
                raw = {'Y': [100 if v else 0 for v in visibility], 'Visibility': visibility}
                self.assertEqual(generate_inpaint_mask(raw, seq_len=seq_len), expected)
        self.assertEqual(generate_inpaint_mask({'Y': [30, 0, 100], 'Visibility': [1, 0, 1]}),
                         [0, 0, 0])
        self.assertEqual(generate_inpaint_mask({'Frame': [100, 102, 103], 'Y': [100, 0, 100],
                                               'Visibility': [1, 0, 1]}, seq_len=4), [0, 0, 0])

    def test_ensemble_weights_preserve_existing_convention(self):
        torch.testing.assert_close(get_ensemble_weight(4, 'weight'),
                                   torch.tensor([1., 2., 2., 1.]) / 6)
        torch.testing.assert_close(get_ensemble_weight(3, 'average'), torch.ones(3) / 3)
        torch.testing.assert_close(get_ensemble_weight(1, 'weight'), torch.ones(1))

    def test_all_missing_and_empty_input_never_call_model(self):
        for mode in MODES:
            for length in (0, 1, 9):
                with self.subTest(mode=mode, length=length):
                    coordinates, mask = trajectory([0] * length)
                    model = ConstantInpaint()
                    result, eligible = inpaint_trajectory(model, coordinates, mask, 4, eval_mode=mode)
                    self.assertTrue(torch.equal(result, coordinates))
                    self.assertEqual(eligible.shape, (length, 1))
                    self.assertFalse(eligible.any())
                    self.assertEqual(model.windows_seen, 0)

    def test_leading_and_trailing_gaps_are_not_extrapolated(self):
        coordinates, mask = trajectory([0, 0, 1, 0, 1, 0, 0])
        for mode in MODES:
            with self.subTest(mode=mode):
                result, eligible = inpaint_trajectory(ConstantInpaint(), coordinates, mask, 5,
                                                     eval_mode=mode)
                self.assertEqual(eligible[:, 0].tolist(), [False, False, False, True, False, False, False])
                self.assertTrue(torch.equal(result[[0, 1, 5, 6]], torch.zeros((4, 2))))
                self.assertTrue(torch.equal(result[3], torch.full((2,), 0.5)))

    def test_exact_context_limit_rejects_the_whole_overlong_gap(self):
        for mode in MODES:
            for gap in (2, 3, 8):
                with self.subTest(mode=mode, gap=gap):
                    coordinates, mask = trajectory([1] + [0] * gap + [1])
                    result, eligible = inpaint_trajectory(ConstantInpaint(), coordinates, mask, 4,
                                                         eval_mode=mode)
                    self.assertEqual(int(eligible.sum()), gap if gap == 2 else 0)
                    expected = 0.5 if gap == 2 else 0.
                    self.assertTrue(torch.equal(result[1:-1], torch.full((gap, 2), expected)))
        coordinates, mask = trajectory([1, 0, 1])
        for seq_len in (1, 2):
            for mode in MODES:
                with self.subTest(seq_len=seq_len, mode=mode):
                    result, eligible = inpaint_trajectory(ConstantInpaint(), coordinates, mask, seq_len,
                                                         eval_mode=mode)
                    self.assertTrue(torch.equal(result, coordinates))
                    self.assertFalse(eligible.any())

    def test_cross_window_gap_uses_only_supported_windows(self):
        coordinates, mask = trajectory([1, 1, 1, 0, 0, 1, 1, 1])
        for mode in MODES:
            with self.subTest(mode=mode):
                result, eligible = inpaint_trajectory(ConstantInpaint(), coordinates, mask, 4,
                                                     eval_mode=mode, batch_size=2)
                expected = 0. if mode == 'nonoverlap' else 0.5
                self.assertTrue(torch.equal(result[3:5], torch.full((2, 2), expected)))
                self.assertEqual(int(eligible.sum()), 0 if mode == 'nonoverlap' else 2)

    def test_observations_and_inputs_survive_masking_and_thresholding(self):
        coordinates = torch.tensor([[0.001, 0.002], [0., 0.], [0., 0.5]])
        original = coordinates.clone()
        for mode in MODES:
            with self.subTest(mode=mode):
                result, eligible = inpaint_trajectory(
                    ConstantInpaint(0.01), coordinates, torch.ones((3, 1)), 4,
                    eval_mode=mode, coor_th=COOR_TH)
                self.assertTrue(torch.equal(result[[0, 2]], original[[0, 2]]))
                self.assertTrue(torch.equal(result[1], torch.zeros(2)))
                self.assertEqual(eligible[:, 0].tolist(), [False, True, False])
                self.assertTrue(torch.equal(coordinates, original))

    def test_short_input_padding_does_not_add_frames_or_evidence(self):
        coordinates, mask = trajectory([1, 0, 1])
        for mode in MODES:
            with self.subTest(mode=mode):
                result, eligible = inpaint_trajectory(ConstantInpaint(), coordinates, mask, 16,
                                                     eval_mode=mode)
                self.assertEqual(result.shape, (3, 2))
                self.assertEqual(eligible[:, 0].tolist(), [False, True, False])
                self.assertTrue(torch.equal(result[1], torch.full((2,), 0.5)))
        for visibility in ([1], [1, 0], [0, 1]):
            coordinates, mask = trajectory(visibility)
            result, eligible = inpaint_trajectory(ConstantInpaint(), coordinates, mask, 16)
            self.assertTrue(torch.equal(result, coordinates))
            self.assertFalse(eligible.any())

    def test_candidate_context_is_retained_without_accepting_unsupported_repairs(self):
        coordinates, mask = trajectory([0, 1, 0, 1, 0])
        model = ConstantInpaint()
        with patch.object(model, 'forward', wraps=model.forward) as forward:
            result, eligible = inpaint_batch(model, coordinates[None], mask[None])
        self.assertTrue(torch.equal(forward.call_args[0][1], mask[None]))
        self.assertEqual(eligible[0, :, 0].tolist(), [False, False, True, False, False])
        self.assertTrue(torch.equal(result[0, [0, 4]], torch.zeros((2, 2))))

    def test_frame_discontinuities_and_rally_boundaries_are_not_bridged(self):
        coordinates, mask = trajectory([1, 0, 0, 1])
        for indices in (
                [[0, 100], [0, 101], [0, 103], [0, 104]],
                [[0, 100], [0, 101], [1, 102], [1, 103]]):
            with self.subTest(indices=indices):
                model = ConstantInpaint()
                result, eligible = inpaint_trajectory(model, coordinates, mask, 4, indices=indices)
                self.assertTrue(torch.equal(result, coordinates))
                self.assertFalse(eligible.any())
                self.assertEqual(model.windows_seen, 0)

    def test_supported_ensemble_weights_are_renormalized(self):
        coordinates, mask = trajectory([1, 1, 1, 1, 1, 0, 1, 1, 1, 1])
        expected = {'average': 29 / 108, 'weight': 67 / 252}
        for mode, value in expected.items():
            for batch_size in (1, 3, 16):
                with self.subTest(mode=mode, batch_size=batch_size):
                    result, _ = inpaint_trajectory(PositionInpaint(), coordinates, mask, 5,
                                                  eval_mode=mode, batch_size=batch_size)
                    self.assertAlmostEqual(float(result[5, 0]), value, places=6)

    def test_boundary_ensemble_remains_uniform(self):
        coordinates, mask = trajectory([1, 1, 0, 1, 1, 1, 1, 1])
        for mode in ('average', 'weight'):
            with self.subTest(mode=mode):
                result, _ = inpaint_trajectory(PositionInpaint(), coordinates, mask, 5, eval_mode=mode)
                self.assertAlmostEqual(float(result[2, 0]), 13 / 72, places=6)

    def test_final_threshold_applies_only_to_repaired_points(self):
        coordinates, mask = trajectory([1, 1, 1, 0, 1, 1, 1, 1])
        result, eligible = inpaint_trajectory(AlternatingInpaint(), coordinates, mask, 4,
                                             coor_th=COOR_TH)
        self.assertTrue(eligible[3])
        self.assertTrue(torch.equal(result[3], torch.zeros(2)))
        self.assertTrue(torch.equal(result[mask[:, 0] == 0], coordinates[mask[:, 0] == 0]))

    def test_invalid_inputs_and_model_output_raise_explicit_errors(self):
        coordinates, mask = trajectory([1, 0, 1])
        for value in (float('nan'), float('inf'), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                inpaint_batch(ConstantInpaint(value), coordinates[None], mask[None])
        with self.assertRaises(ValueError):
            inpaint_batch(ConstantInpaint(), coordinates[None], torch.ones((1, 3, 2)))
        with self.assertRaises(ValueError):
            inpaint_batch(ConstantInpaint(), coordinates[None], torch.full((1, 3, 1), 2))
        with self.assertRaises(ValueError):
            inpaint_batch(ConstantInpaint(), coordinates[None].long(), mask[None])
        model = ConstantInpaint()
        with patch.object(model, 'forward', return_value=torch.zeros((1, 2))), self.assertRaises(ValueError):
            inpaint_batch(model, coordinates[None], mask[None])
        with self.assertRaises(ValueError):
            inpaint_trajectory(ConstantInpaint(), coordinates, mask, 0)
        with self.assertRaises(ValueError):
            inpaint_trajectory(ConstantInpaint(), coordinates, mask, 4, eval_mode='invalid')


class OutputAndEvaluationTests(unittest.TestCase):
    def test_provenance_preserves_raw_pixels_and_frame_ids(self):
        raw = {'Frame': [100, 101, 102], 'X': [1, 0, 0], 'Y': [2, 0, 0], 'Visibility': [1, 0, 0]}
        original = copy.deepcopy(raw)
        result = format_inpaint_predictions(raw, torch.full((3, 2), 0.5),
                                            torch.tensor([[False], [True], [False]]), (WIDTH, HEIGHT))
        self.assertEqual(raw, original)
        self.assertEqual(result['Frame'], raw['Frame'])
        self.assertEqual(result['X'], [1, 256, 0])
        self.assertEqual(result['Y'], [2, 144, 0])
        self.assertEqual(result['Visibility'], [1, 1, 0])
        self.assertEqual(result['Visibility_Raw'], [1, 0, 0])
        self.assertEqual(result['X_Raw'], [1, 0, 0])
        self.assertEqual(result['Inpaint_Mask'], [0, 1, 0])

    def test_default_and_training_csv_schemas_remain_compatible(self):
        raw = {'Frame': [7], 'X': [1], 'Y': [2], 'Visibility': [1]}
        pred = format_inpaint_predictions(raw, torch.zeros((1, 2)), torch.zeros((1, 1)), (WIDTH, HEIGHT))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'prediction.csv')
            write_pred_csv(pred, path)
            self.assertEqual(pd.read_csv(path).columns.tolist(), ['Frame', 'Visibility', 'X', 'Y'])
            write_pred_csv(pred, path, save_provenance=True)
            self.assertEqual(pd.read_csv(path).columns.tolist(),
                             ['Frame', 'Visibility', 'X', 'Y', 'Visibility_Raw', 'X_Raw', 'Y_Raw', 'Inpaint_Mask'])
            pred.update({'Visibility_GT': [1], 'X_GT': [1], 'Y_GT': [2]})
            write_pred_csv(pred, path, save_inpaint_mask=True)
            self.assertEqual(pd.read_csv(path).columns.tolist(),
                             ['Frame', 'Visibility_GT', 'X_GT', 'Y_GT', 'Visibility', 'X', 'Y', 'Inpaint_Mask'])

    def test_linear_interpolation_uses_interior_positions_without_extrapolation(self):
        np.testing.assert_allclose(evaluation.linear_interp([10, 0, 0, 40], [0, 1, 1, 0]),
                                   [10, 20, 30, 40])
        np.testing.assert_allclose(evaluation.linear_interp([10, 0, 30], [0, 1, 0]), [10, 20, 30])
        np.testing.assert_allclose(evaluation.linear_interp([0, 0, 30], [1, 1, 0]), [0, 0, 30])
        np.testing.assert_allclose(evaluation.linear_interp([10, 0, 0], [0, 1, 1]), [10, 0, 0])
        with self.assertRaises(ValueError):
            evaluation.linear_interp([10, 0, 30], [0, 2, 0])

    def test_coordinate_evaluation_does_not_mutate_tensors_or_arrays(self):
        for convert in (torch.tensor, np.array):
            with self.subTest(convert=convert):
                coordinates = convert([[[0.25, 0.5]], [[0., 0.]]])
                array = coordinates.numpy() if torch.is_tensor(coordinates) else coordinates
                original = array.copy()
                indices = np.array([[[0, 100]], [[0, 101]]])
                first = evaluation.evaluate(indices, c_true=coordinates, c_pred=coordinates)
                second = evaluation.evaluate(indices, c_true=coordinates, c_pred=coordinates)
                self.assertEqual(first, second)
                self.assertEqual(first['Frame'], [100, 101])
                np.testing.assert_array_equal(array, original)

    def test_training_evaluation_rejects_fully_masked_windows(self):
        coordinates = torch.zeros((1, 4, 2))
        truth = torch.full_like(coordinates, 0.5)
        visibility = torch.zeros((1, 4, 1))
        indices = torch.tensor([[[0, i] for i in range(4)]])
        loader = [(indices, coordinates, truth, visibility, torch.ones_like(visibility),
                   torch.ones_like(visibility))]
        model = ConstantInpaint()
        with patch.object(torch.Tensor, 'cuda', lambda value: value):
            loss, result = evaluation.eval_inpaintnet(model, loader, {'verbose': False, 'tolerance': 4})
        self.assertEqual(model.windows_seen, 0)
        self.assertEqual(loss, 0.)
        self.assertEqual(result['inpaint']['FN'], 4)
        self.assertEqual(result['inpaint']['FP2'], 0)
        self.assertEqual(result['reconstruct']['TN'], 4)

    def test_training_evaluation_preserves_supported_repairs_and_baseline(self):
        coordinates, mask = trajectory([1, 0, 1])
        truth = coordinates.clone()
        truth[1] = 0.5
        visibility = (coordinates != 0).any(dim=-1)[None, :, None].float()
        indices = torch.tensor([[[0, 100], [0, 101], [0, 102]]])
        loader = [(indices, coordinates[None], truth[None], visibility, torch.ones_like(visibility),
                   mask[None])]
        with patch.object(torch.Tensor, 'cuda', lambda value: value):
            loss, result = evaluation.eval_inpaintnet(ConstantInpaint(), loader,
                                                      {'verbose': False, 'tolerance': 4})
        self.assertEqual(loss, 0.)
        self.assertEqual(result['inpaint']['TP'], 3)
        self.assertEqual(result['baseline']['TP'], 2)
        self.assertEqual(result['baseline']['FN'], 1)

    def test_rally_evaluation_uses_shared_policy_and_preserves_frame_ids(self):
        for mode in MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                match_dir = os.path.join(directory, 'match1')
                rally_dir = os.path.join(match_dir, 'frame', 'rally')
                os.makedirs(rally_dir)
                os.makedirs(os.path.join(match_dir, 'predicted_csv'))
                Image.new('RGB', (WIDTH * 2, HEIGHT * 2)).save(os.path.join(rally_dir, '0.png'))
                points = np.array(POINTS)
                truth = points.copy()
                truth[2] = (256, 144)
                visibility = (points != 0).any(axis=-1).astype(int).tolist()
                pd.DataFrame({
                    'Frame': list(range(100, 109)), 'X': points[:, 0], 'Y': points[:, 1],
                    'Visibility': visibility, 'X_GT': truth[:, 0], 'Y_GT': truth[:, 1],
                    'Visibility_GT': (truth != 0).any(axis=-1).astype(int),
                    'Inpaint_Mask': [0, 0, 1, 0, 1, 1, 1, 0, 1],
                }).to_csv(os.path.join(match_dir, 'predicted_csv', 'rally_ball.csv'), index=False)
                result = evaluation.test_rally((None, ConstantInpaint()), rally_dir,
                                                {'inpaintnet_seq_len': 4, 'eval_mode': mode,
                                                 'batch_size': 2, 'tolerance': 4})
                self.assertEqual(result['Frame'], list(range(100, 109)))
                self.assertEqual(result['X'][0], 2)
                self.assertEqual(result['Y'][0], 4)
                self.assertEqual(result['X'][2], 512)
                self.assertEqual(result['Y'][2], 288)
                self.assertEqual(result['Visibility_Raw'], visibility)
                self.assertEqual(result['Inpaint_Mask'], [0, 0, 1, 0, 0, 0, 0, 0, 0])
                self.assertEqual(result['Visibility'], [1, 1, 1, 1, 0, 0, 0, 1, 0])
                self.assertEqual(result['Type'], [0, 0, 0, 0, 1, 1, 1, 0, 1])
                if mode == 'nonoverlap':
                    path = os.path.join(match_dir, 'predicted_csv', 'rally_ball.csv')
                    invalid = pd.read_csv(path)
                    invalid['Inpaint_Mask'] = invalid['Inpaint_Mask'].astype(float)
                    invalid.loc[2, 'Inpaint_Mask'] = 0.5
                    invalid.to_csv(path, index=False)
                    with self.assertRaisesRegex(ValueError, 'Stored Inpaint_Mask'):
                        evaluation.test_rally((None, ConstantInpaint()), rally_dir,
                                              {'inpaintnet_seq_len': 4, 'eval_mode': mode,
                                               'batch_size': 2, 'tolerance': 4})

    def run_predict_cli(self, mode, provenance, with_inpaint=True):
        tracker, inpainter = TrackingStub(), ConstantInpaint()
        frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(len(POINTS))]

        def checkpoint(path, **kwargs):
            if path == 'tracknet.pt':
                return {'param_dict': {'seq_len': 2, 'bg_mode': ''}, 'model': tracker.state_dict()}
            return {'param_dict': {'seq_len': 4}, 'model': inpainter.state_dict()}

        def data_loader(dataset, **kwargs):
            kwargs['num_workers'] = 0
            return DataLoader(dataset, **kwargs)

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            argv = ['predict.py', '--video_file', 'sample.mp4', '--tracknet_file', 'tracknet.pt',
                    '--save_dir', directory, '--eval_mode', mode, '--batch_size', '2']
            if with_inpaint:
                argv.extend(['--inpaintnet_file', 'inpaintnet.pt'])
            if provenance:
                argv.append('--output_provenance')
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch('torch.load', side_effect=checkpoint))
            stack.enter_context(patch('torch.cuda.is_available', return_value=False))
            stack.enter_context(patch.object(torch.backends, 'mps',
                                            types.SimpleNamespace(is_available=lambda: False), create=True))
            stack.enter_context(patch('torch.utils.data.DataLoader', side_effect=data_loader))
            stack.enter_context(patch('tqdm.tqdm', side_effect=lambda values: values))
            stack.enter_context(patch('utils.general.generate_frames', return_value=frames))
            stack.enter_context(patch('utils.general.get_model',
                                      side_effect=lambda name, *args: tracker if name == 'TrackNet' else inpainter))
            capture = stack.enter_context(patch('utils.general.cv2.VideoCapture'))
            capture.return_value.get.side_effect = lambda prop: {3: WIDTH, 4: HEIGHT}.get(prop, 0)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            runpy.run_path(str(Path(__file__).resolve().parents[1] / 'predict.py'), run_name='__main__')
            return pd.read_csv(os.path.join(directory, 'sample_ball.csv'))

    def test_prediction_cli_modes_and_opt_in_schema_with_model_stubs(self):
        for mode in MODES:
            for provenance in (False, True):
                with self.subTest(mode=mode, provenance=provenance):
                    result = self.run_predict_cli(mode, provenance)
                    self.assertEqual(result['Frame'].tolist(), list(range(len(POINTS))))
                    self.assertEqual(result['X'].tolist(), [1, 60, 256, 120, 0, 0, 0, 200, 0])
                    self.assertEqual(result['Y'].tolist(), [2, 80, 144, 100, 0, 0, 0, 120, 0])
                    self.assertEqual(result['Visibility'].tolist(), [1, 1, 1, 1, 0, 0, 0, 1, 0])
                    if provenance:
                        self.assertEqual(result['X_Raw'].tolist(), [point[0] for point in POINTS])
                        self.assertEqual(result['Inpaint_Mask'].tolist(), [0, 0, 1, 0, 0, 0, 0, 0, 0])
                    else:
                        self.assertEqual(result.columns.tolist(), ['Frame', 'Visibility', 'X', 'Y'])

    def test_tracknet_only_cli_provenance_does_not_claim_repairs(self):
        result = self.run_predict_cli('nonoverlap', True, with_inpaint=False)
        self.assertEqual(result['X'].tolist(), result['X_Raw'].tolist())
        self.assertEqual(result['Y'].tolist(), result['Y_Raw'].tolist())
        self.assertEqual(result['Visibility'].tolist(), result['Visibility_Raw'].tolist())
        self.assertEqual(result['Inpaint_Mask'].tolist(), [0] * len(POINTS))


if __name__ == '__main__':
    unittest.main()
