# TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification
We present TrackNetV3, a model composed of two core modules: trajectory prediction and rectification. The trajectory prediction module leverages an estimated background as auxiliary data to locate the shuttlecock in spite of the fluctuating visual interferences. This module also incorporates mixup data augmentation to formulate complex
scenarios to strengthen the network’s robustness. Given that a shuttlecock can occasionally be obstructed, we create repair masks by analyzing the predicted trajectory, subsequently rectifying the path via inpainting.
[[paper](https://dl.acm.org/doi/10.1145/3595916.3626370)]

<div align="center">
    <a href="./">
        <img src="./figure/NetArch.png" width="50%"/>
    </a>
</div>

## Performance 

* Performance on the test split of [Shuttlecock Trajectory Dataset](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw).
* These are the original benchmark results, not measurements of the conservative repair policy described below.

<div align="center">
    <table>
    <thead>
        <tr>
        <th>Model</th> <th>Accuracy</th> <th>Precision</th> <th>Recall</th> <th>F1</th> <th>FPS</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>YOLOv7</td> <td>57.82%</td> <td>78.53%</td> <td>59.96%</td> <td>68.00%</td> <td><b>34.77</b></td>
        </tr>
        <tr>
        <td>TrackNetV2</td> <td>94.98%</td> <td><b>99.64%</b></td> <td>94.56%</td> <td>97.03%</td> <td>27.70</td>
        </tr>
        <tr>
        <td>TrackNetV3</td> <td><b>97.51%</b></td> <td>97.79%</td> <td><b>99.33%</b></td> <td><b>98.56%</b></td> <td>25.11</td>
        </tr>
    </tbody>
    </table>
    </br>
    <a href="./">
        <img src="./figure/Comparison.png" width="80%"/>
    </a>
</div>

## Installation
* Develop Environment
    ```
    Ubuntu 16.04.7 LTS
    Python 3.8.7
    torch 1.10.0
    ```
* Clone this reposity.
    ```
    git clone https://github.com/tloy1966/TrackNetV3.git
    ```

* Install the requirements.
    ```
    pip install -r requirements.txt
    ```

## Inference
* Download the [checkpoints](https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing)
* Unzip the file and place the parameter files to ```ckpts```
    ```
    unzip TrackNetV3_ckpts.zip
    ```
* Predict the label csv from the video
    ```
    python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction
    ```
* Predict the label csv from the video, and output a video with predicted trajectory
    ```
    python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction --output_video
    ```
* For large video
    * Enable ```--large_video``` to cache a sliding window of preprocessed frames instead of holding all decoded frames. This does not bound total process/GPU memory or guarantee a speedup; background sampling, batches, temporal ensembling, and output trajectories also consume memory.
    * Use ```--max_sample_num``` to set the number of samples for background estimation.
    * Use ```--video_range``` to specify the start and end seconds of the video for background estimation.
    ```
    python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction --large_video --video_range 324,330
    ```

### Conservative trajectory repair

InpaintNet repairs a missing run only when both bounding TrackNet detections are present in the **same contiguous input window**. For a checkpoint with sequence length `L`, a gap can therefore contain at most `L - 2` missing frames. The existing endpoint-height heuristic also applies. Leading/trailing gaps, fully missing windows, and longer gaps stay missing: `X=0, Y=0, Visibility=0`. Repaired points never become evidence for further repairs, and raw detections are preserved, including points near the coordinate origin.

This rule applies to video prediction (including `--large_video`), rally evaluation, and training-time InpaintNet evaluation. In `nonoverlap` mode, even a short gap crossing a window boundary can remain missing if neither window contains both endpoints. The `average` and `weight` modes can use a spanning window; unsupported windows are excluded and the remaining ensemble weights are renormalized. Input padding for trajectories shorter than `L` does not add output frames or observations.

These are conservative eligibility checks, not an accuracy guarantee: incorrect raw detections can still provide incorrect context. No retraining, real-video accuracy measurement, or speed benchmark is included in this change.

### Optional output provenance

The default prediction CSV remains `Frame,Visibility,X,Y`. Add `--output_provenance` to append `Visibility_Raw,X_Raw,Y_Raw,Inpaint_Mask` without requiring ground-truth labels:

```
python predict.py --video_file test.mp4 --tracknet_file ckpts\TrackNet_best.pt --inpaintnet_file ckpts\InpaintNet_best.pt --save_dir prediction --output_provenance
```

| Columns | Meaning |
| --- | --- |
| `Frame` | Original frame index; repair does not renumber or add frames. |
| `Visibility,X,Y` | Final prediction, using zero visibility and zero coordinates for missing points. |
| `Visibility_Raw,X_Raw,Y_Raw` | TrackNet detections before repair, in the same pixel scale as the final output. |
| `Inpaint_Mask` | `1` if this missing point was eligible for repair in at least one contributing window, otherwise `0`. |

A visible point with `Visibility_Raw=0` and `Inpaint_Mask=1` is repaired, not directly detected. An eligible point can still remain missing if the model output is suppressed by the coordinate threshold. Without an InpaintNet checkpoint, provenance contains the unchanged raw detections and a zero repair mask. Neither visibility nor the repair mask is a calibrated confidence score; the optional evaluation heatmap score is not calibrated either.

`test.py --video_file ... --output_provenance` supports the same additive CSV columns. For dataset evaluation, use `--output_pred` to retain per-frame JSON details, including provenance for repaired trajectories. The training-data CSV format produced by `generate_mask_data.py` is unchanged and still includes ground truth; its stored masks are candidates that are checked again against each inference window.

### Configurable heatmap decoding

`--heatmap_threshold` defaults to `0.5` and accepts finite values in `[0,1]`. Decoding uses **strictly greater than** the threshold, so a value exactly equal to `0.5` is still rejected at the default. The setting is shared by prediction, labeled/training evaluation, mask-data generation, and training visualizations. It does not change the training loss, coordinate suppression (`COOR_TH`), visibility/mask meanings, or confidence calibration.

Use the same labeled inputs and protocol to compare thresholds; there is no automatic threshold search or accuracy claim for any new value.

### Reproducible run artifacts

Add `--output_run` to save a JSON sidecar with effective settings, checkpoint SHA-256 identities and stored parameters, implementation-file hashes, runtime/device versions, video metadata, per-video results, aggregate metrics, and synchronized stage timings. Prediction writes `<video_name>_run.json` next to the ordinary CSV. No new sidecar or CSV columns are enabled by default.

Checkpoint identity is captured from the same open file used for loading, before video inference starts. A checkpoint that changes during loading/hashing is rejected rather than silently recording newer weights under an older run.

Without labels, reports contain only observable statistics such as visible/missing frames, eligible/repaired points, actual frame/window counts, and timings. The `metrics` field is `null`; no accuracy, precision, recall, or F1 is inferred from how many points were repaired.

For labeled video prediction, supply `--label_file` with `Frame,Visibility,X,Y` in original video pixels. It requires `--output_run` and exactly one label row for every output frame ID; rows are aligned by ID, not their order.

```
python predict.py --video_file test.mp4 --tracknet_file ckpts\TrackNet_best.pt --inpaintnet_file ckpts\InpaintNet_best.pt --large_video --output_run --label_file labels.csv --heatmap_threshold 0.5 --tolerance 4 --save_dir runs\threshold_05
```

Repeat with another threshold and a separate `--save_dir` to retain both runs. `videos[].metrics.raw` and `videos[].metrics.final` compare the raw TrackNet and repaired trajectories against the **same ground truth**; an extra repaired false positive is not an accuracy improvement.

| Report field | Contents |
| --- | --- |
| `settings`, `models`, `implementation_sha256`, `runtime` | Effective configuration, checkpoint/content identity, implementation identity, software/device/thread settings. |
| `evaluation_protocol` | Coordinate space, tolerance, quantization, metric formulas, and any effective-trajectory filter identity. |
| `videos[].statistics` | Raw/final visibility and missing-run counts, eligible repairs, and repaired visible points. |
| `videos[].metrics`, `metrics` | Per-video and aggregate raw/final GT-based metrics, or `null` without GT. |
| `videos[].timing`, `timing` | Per-video/run timings, original-frame throughput, and separate window/padded-model-input counts. |

Distances retain the repository's coordinate-evaluation convention: normalized coordinates are mapped to `512x288` input pixels and each coordinate is truncated before Euclidean distance is compared with `--tolerance` (`<=` is a match). Tolerance is **not** in original-video pixels. The existing TP/TN/FP1/FP2/FN definitions and metric formulas are recorded, not replaced by a different detection protocol.

`test.py --output_run` writes `<split>_run_<eval_mode>.json`; its per-video GT metrics use the same effective row ranges as the existing test-split metrics, while throughput counts all original processed frames. Video evaluation writes `<rally_id>_run.json`. With both checkpoints supplied, evaluation now actually reruns TrackNet and repairs those fresh detections instead of ignoring the supplied TrackNet model in favor of a cache.

With only `--inpaintnet_file`, evaluation still reads `predicted_csv` coordinates and the `*_GT` labels stored there. A newly requested heatmap threshold cannot change cached coordinates. Generate caches with `generate_mask_data.py --output_run --heatmap_threshold ...` to write matching `<csv_file>.run.json` identities and a `mask_generation_run_<eval_mode>.json` in the data root. Evaluation verifies the CSV hash before trusting that generation metadata. Legacy caches without a sidecar remain usable, but their generation threshold/model are explicitly unknown; regenerate caches after changing labels to avoid stale cached GT.

### Streaming, background sampling, and timing

The streaming path converts BGR to RGB, resizes, applies the existing background/channel operations, and normalizes each real frame once. At most `L` processed `float32` frames are cached. Channel order, original-resolution subtraction, padding IDs, output scaling, temporal weights, and conservative repair are preserved. Empty/exact-boundary videos do not produce extra windows; short videos emit each real frame once. Short overlapping clips now average only actual votes, correcting the old tail denominator that could count nonexistent windows and suppress heatmaps.

Background estimation still computes the exact NumPy median of selected **original-resolution** samples; it is not replaced by a resized or approximate median. Sampling is deterministic with a hard `--max_sample_num` limit: stride is `ceil(segment_frames / max_sample_num)` (at least one). `--video_range start,end` is a `[start,end)` time interval, using the video's fractional FPS and ceiling frame boundaries. Actual indices and sample-buffer/median sizes are saved in streaming run metadata.

The old floor-stride sampler could exceed its stated maximum, and truncated FPS could include frames outside a requested range. Fixing those cases can intentionally change the selected samples and therefore the background. Compare runs with the same recorded sampling policy/range. A lower `--max_sample_num` is an explicit background-quality/memory tradeoff; even the default can require substantial RAM for high-resolution videos.

`peak_cache_bytes` measures only the processed-frame cache, not RSS, decoder buffers, median workspace, model/batch/ensemble tensors, or retained output trajectories. Retained output trajectories still grow with video length. The non-streaming path still preloads all frames and repeats window preprocessing; its background is the all-frame median and does not use the streaming sampling controls. Preprocessing equivalence is checked using the same median, not by claiming different sampling policies produce identical backgrounds.

Timing is opt-in through `--output_run`. CUDA/MPS stages synchronize before and after accelerator work. Model-only seconds include TrackNet and eligible InpaintNet forward calls, excluding transfers; both FPS numerators count original frames, never overlapping windows or padding. End-to-end time includes checkpoint setup/hash work, input/background preparation, decode, preprocessing, transfers, inference, postprocessing/metrics, and requested CSV/video outputs. Each report excludes its own final metadata assembly/serialization; aggregate mask-generation timing also includes earlier per-cache sidecar work.

Stage times can be nested and must not be summed. Streaming decode/preprocess/background stages are measured directly; `input` also includes DataLoader wait/collation. For worker-backed non-streaming/rally input it is wait time, not summed worker CPU time. No warmup is performed, first forwards and profiling synchronization overhead are included, and per-rally timing excludes reused checkpoint loading while aggregate timing includes it. These definitions are not a measured real-video accuracy/FPS improvement.

## Training
### 1. Prepare Dataset
* Download [Shuttlecock Trajectory Dataset](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw)
* Adjust file structure:
    1. Merge the `Professional` and `Amateur` match directories into a single `train` directory.
    2. Rename the `Amateur` match directories to start from `match24` through `match26`.
    3. Rename the `Test` directory to `test`.
* Dataset file structure:
```
  data
    ├─ train
    |   ├── match1/
    |   │   ├── csv/
    |   │   │   ├── 1_01_00_ball.csv
    |   │   │   ├── 1_02_00_ball.csv
    |   │   │   ├── …
    |   │   │   └── *_**_**_ball.csv
    |   │   ├── frame/
    |   │   │   ├── 1_01_00/
    |   │   │   │   ├── 0.png
    |   │   │   │   ├── 1.png
    |   │   │   │   ├── …
    |   │   │   │   └── *.png
    |   │   │   ├── 1_02_00/
    |   │   │   │   ├── 0.png
    |   │   │   │   ├── 1.png
    |   │   │   │   ├── …
    |   │   │   │   └── *.png
    |   │   │   ├── …
    |   │   │   └── *_**_**/
    |   │   │
    |   │   └── video/
    |   │       ├── 1_01_00.mp4
    |   │       ├── 1_02_00.mp4
    |   │       ├── …
    |   │       └── *_**_**.mp4
    |   ├── match2/
    |   │ ⋮
    |   └── match26/
    ├─ val
    |   ├── match1/
    |   ├── match2/
    |   │ ⋮
    |   └── match26/
    └─ test
        ├── match1/
        ├── match2/
        └── match3/
```
* Attributes in each csv files: `Frame, Visibility, X, Y`
* Data preprocessing
    ```
    python preprocess.py
    ```
* The `frame` directories and the `val` directory will be generated after preprocessing.
* Check the estimated background images in `<data_dir>/median`
    * If available, the dataset will use the median image of the match; otherwise, it will use the median image of the rally.
    * For example, you can exclude `train/match16/median.npz` due to camera angle discrepancies; therefore, the dataset will resort to the median image of the rally within match 16.
* Set the data root directory to `data_dir` in `dataset.py`.
    * `dataset.py` will generate the image mapping for each sample and cache the result in `.npy` files.
    * If you modify any related functions in `dataset.py`, please ensure you delete these cached files.
### 2. Train Tracking Module
* Train the tracking module from scratch
    ```
    python train.py --model_name TrackNet --seq_len 8 --epochs 30 --batch_size 10 --bg_mode concat --alpha 0.5 --save_dir exp --verbose
    ```

* Resume training (start from the last epoch to the specified epoch)
    ```
    python train.py --model_name TrackNet --epochs 30 --save_dir exp --resume_training --verbose
    ```

### 3. Generate Predicted Trajectories and Inpainting Masks
* Generate predicted trajectories and inpainting masks for training rectification module
    * Noted that the coordinate range corresponds to the input spatial dimensions, not the size of the original image.
    ```
    python generate_mask_data.py --tracknet_file ckpts/TrackNet_best.pt --batch_size 16
    ```

### 4. Train Rectification Module
* Train the rectification module from scratch.
    ```
    python train.py --model_name InpaintNet --seq_len 16 --epoch 300 --batch_size 32 --lr_scheduler StepLR --mask_ratio 0.3 --save_dir exp --verbose
    ```

* Resume training (start from the last epoch to the specified epoch)
    ```
    python train.py --model_name InpaintNet --epochs 30 --save_dir exp --resume_training
    ```

## Evaluation
* Evaluate TrackNetV3 on test set
    ```
    python generate_mask_data.py --tracknet_file ckpts/TrackNet_best.pt --split_list test
    python test.py --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir eval
    ```

* Evaluate the tracking module on test set
    ```
    python test.py --tracknet_file ckpts/TrackNet_best.pt --save_dir eval
    ```

* Generate video with ground truth label and predicted result
    ```
    python test.py --tracknet_file ckpts/TrackNet_best.pt --video_file data/test/match1/video/1_05_02.mp4 
    ```

* The `--linear_interp` comparison uses bounded interpolation, not leading/trailing extrapolation. Its maximum gap is the InpaintNet context length minus two, or the TrackNet context length minus two when no InpaintNet checkpoint is supplied.

### Repair policy regression tests

With the project's Python dependencies installed, run:

```
python -m unittest discover -s tests -p "test_*.py"
```

The tests preserve the original 23 inpainting regressions and add threshold bounds/defaults/propagation, run identities and raw/final metrics, timing/count semantics, streaming preprocessing/cache limits, exact background sampling, short/empty/window-boundary cases, and synthetic MP4/CSV evaluation integration. Model stubs, temporary inputs, and the installed OpenCV codec are used; codec-dependent tests explicitly skip if an MP4 writer is unavailable. No user video, trained checkpoint, GPU, or retraining is required.

Frame-preprocessing equivalence is checked against the existing non-streaming implementation after its float32 conversion (`atol=1e-7`); temporal results are compared with the previous ensemble (`rtol=1e-6, atol=1e-7`) where its boundary denominators are valid. Separate short-clip tests require actual-vote averaging rather than preserving that old bug. A representative labeled video and actual checkpoints are still needed to assess real tracking quality or FPS.

## Error Analysis Interface
* Evaluate TrackNetV3 on test set and save the detail results for error analysis
    ```
    python test.py --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir eval --output_pred
    ```

* Add json path of evaluation results to the file list in `error_analysis.py`
    ```
    30  # Evaluation result file list
    31  if split == 'train':
    32      eval_file_list = [
    33          {'label': label_name, 'value': json_path},
     ⋮                              ⋮
            ]
        elif split == 'val':
            eval_file_list = [
                {'label': label_name, 'value': json_path},
                                    ⋮
            ]
        elif split == 'test':
            eval_file_list = [
                {'label': label_name, 'value': json_path},
                                    ⋮
            ]
        else:
            raise ValueError(f'Invalid split: {split}')                                  
    ```

* Run Dash application
    ```
    python error_analysis.py --split test --host 127.0.0.1
    ```
<div align="center">
    <a href="./">
        <img src="./figure/ErrorAnalysisUI.png" width="70%"/>
    </a>
</div>

## Reference
* TrackNetV2: https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2
* Shuttlecock Trajectory Dataset: https://hackmd.io/@TUIK/rJkRW54cU
* Labeling Tool: https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2?tab=readme-ov-file#label

## License
This project, including both the codebase and the pretrained model checkpoints (e.g., weights hosted on Google Drive), is released under the [MIT License](./LICENSE). 

You are free to use, modify, and distribute it for both non-commercial and commercial purposes.