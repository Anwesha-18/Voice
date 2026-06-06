# Sign Language Recognition Optimization Report

## Executive Summary

This report documents the current VOICE pipeline and identifies the most effective optimization opportunities for latency, accuracy, and robustness.

The system is already optimized in several key areas:
- browser-side capture at up to 40 FPS,
- server-side MediaPipe frame skipping,
- per-client Holistic tracking,
- optional TFLite inference when available.

The primary focus here is to describe what is implemented, what is measured, and where the next improvements should be made.

---

## Current system state

### Pipeline overview

1. Browser captures webcam frames at `CAPTURE_INTERVAL_MS = 25` ms.
2. Frames are downscaled to `320×240`, JPEG-compressed at quality `0.30`, and sent to `/api/predict`.
3. The backend flips the image, converts to RGB, and runs MediaPipe Holistic.
4. Hand landmarks are normalized and concatenated into `126` features.
5. A rolling buffer of `SEQ_LEN = 30` frames is collected per client.
6. Once the buffer is full, the backend runs model inference and returns the best word plus top-3 scores.
7. The frontend appends words to the sentence only when the prediction is stable enough.

### Current thresholds and controls

- Backend confidence threshold: `CONF_THRESHOLD = 0.55`
- Frontend append threshold: `APPEND_THRESHOLD = 0.50`
- Word append persistence: `APPEND_FRAMES = 1`
- Repeat-word cooldown: `600 ms`
- MediaPipe frame skip: `MP_FRAME_SKIP = 3`

### Supported classes

The current data collection and dataset builder scripts are configured for these 12 words:
- `hello`, `yes`, `no`, `stop`, `thank_you`, `help`, `food`, `water`, 'medicine', 'doctor','please','bathroom'

> Note: the codebase contains additional raw folders, but `collect_data.py` and `build_dataset.py` currently use this reduced set.

### Backend inference support

The backend loads one of:
- `outputs/saved_models/best_model.tflite` if available,
- otherwise `outputs/saved_models/best_model.h5`.

This enables faster TensorFlow Lite inference when the model has been converted.

---

## Measured latency characteristics

### Frontend capture and transport

- Frame capture interval: `25 ms` → theoretical 40 FPS.
- Effective frame send time depends on network and backend processing.
- Downscaling to `320×240` and JPEG compression reduces payload size.

### Backend processing stages

- Decode base64 JPEG
- Flip and convert to RGB
- MediaPipe Holistic processing every third frame
- Feature extraction and normalization
- Optional server-side landmark drawing and JPEG encoding
- Model inference via TFLite or TensorFlow Keras

The backend returns timing information for each stage in every `/api/predict` response. The frontend logs these values to the browser console when available.

### Current latency optimizations in code

- `MP_FRAME_SKIP = 3` reduces MediaPipe cost while preserving temporal smoothing.
- Server keeps a per-client Holistic instance for stable gesture history.
- Inference is executed in a dedicated single-worker thread pool to avoid blocking Flask request handling.
- Word output is suppressed for `idle` and low-confidence predictions.
- Sentence generator falls back to heuristic output if Gemini is unavailable.

---

## What is already optimized

### Frontend

- High capture frequency: `25 ms` / 40 FPS possible.
- Small payloads: `320×240` capture resolution + JPEG quality 0.30.
- Single-frame append logic for rapid sentence assembly.

### Backend

- Frame batching with a rolling buffer of 30 frames.
- MediaPipe skip optimization: process every 3rd frame.
- TFLite support for faster inference when available.
- Per-client memory cleanup for stale sessions.

### Model

- Lightweight BiLSTM + attention architecture.
- 126-feature input keeps the model compact.
- Training script adds Gaussian noise and class-weight balancing.

---

## Optimization opportunities

These are the highest-value improvements to pursue next.

### 1. Verify and use TFLite inference

If `outputs/saved_models/best_model.tflite` exists, the backend will use it automatically. If your current model is still H5-only, convert it with `model/convert_tflite.py` and validate end-to-end latency.

Benefit: reduced inference time and lower CPU usage.

### 2. Measure server-side annotation cost

The API supports an optional latency test mode that disables landmark drawing and JPEG encoding.

- If drawing + encoding is a dominant cost, keep `LATENCY_TEST_MODE = true` in `frontend/src/App.jsx` during profiling.
- If this stage is expensive, move annotation to the browser or skip it entirely in production.

### 3. Tune MediaPipe frame skipping adaptively

`MP_FRAME_SKIP = 3` is a good default, but a dynamic policy could give better tradeoffs:
- lower skip when motion is present,
- higher skip during static periods.

Benefit: reduce CPU usage while preserving landmark fidelity.

### 4. Improve sentence stability without slowing recognition

Current frontend settings already favor responsiveness:
- `APPEND_FRAMES = 1`
- `APPEND_THRESHOLD = 0.50`

If false positives appear, increase `APPEND_FRAMES` to `2` or raise `APPEND_THRESHOLD` gradually until the balance is acceptable.

### 5. Expand or regularize gesture classes

The dataset and model are currently limited to 8 configured words. If you want to support more gestures:
- update `WORDS` in `preprocessing/collect_data.py`
- update `WORDS` in `preprocessing/build_dataset.py`
- collect additional sequences for each new class
- retrain the model

Benefit: broader vocabulary and more natural phrase generation.

---

## Recommended testing strategy

### Latency validation

Use the browser console and `/api/metrics` to measure:
- round-trip time for `/api/predict`
- per-stage backend timings
- active client count
- total frames processed

### Accuracy validation

Check the live UI for:
- correct top-1 word selection
- stability of consecutive word appending
- false positives from low-confidence predictions

If accuracy drops, tune `CONF_THRESHOLD` or `APPEND_THRESHOLD` while preserving the fastest acceptable response.

### Resource validation

Monitor CPU and memory usage on the backend machine.
- MediaPipe and TensorFlow are the heaviest components.
- TFLite inference usually reduces CPU compared to H5.

---

## Recommended configuration ranges

| Parameter | Current | Suggested range | Notes |
|-----------|---------|-----------------|-------|
| `CONF_THRESHOLD` | `0.55` | `0.50–0.70` | Lower improves latency; higher improves precision |
| `APPEND_THRESHOLD` | `0.50` | `0.50–0.65` | Affects when the frontend accepts a word |
| `APPEND_FRAMES` | `1` | `1–3` | More frames improves stability, slower response |
| `CAPTURE_INTERVAL_MS` | `25` | `20–40` | Faster capture increases responsiveness but raises CPU/network load |
| `MP_FRAME_SKIP` | `3` | `2–4` | Tradeoff between MediaPipe latency and motion fidelity |
| Back-to-back cooldown | `600 ms` | `400–1000 ms` | Controls repeated word suppression |

---

## Practical recommendations

1. Keep `APPEND_FRAMES` at `1` if you want the fastest recognition path.
2. Use `APPEND_THRESHOLD = 0.50` for aggressive recognition, and increase it if false positives become a problem.
3. If latency is still high, enable `LATENCY_TEST_MODE` and compare timings with and without server-side annotating.
4. Prefer the `.tflite` model in production to reduce inference cost.
5. Add new gestures by extending the configured `WORDS` list and rebuilding the dataset.

---

## Notes on current implementation

- The backend logs detailed timings for every `/api/predict` response.
- The frontend currently implements a single-frame append rule for rapid sentence construction.
- `generate-sentence` uses Gemini when available, otherwise falls back to a deterministic heuristic.
- `dataset/raw_sequences/` contains many folders, but the current pipeline only consumes the 8 configured gestures.

---

## Conclusion

The VOICE system is already optimized around a fast capture-to-prediction loop. The next improvements should focus on:
- verifying TFLite inference,
- measuring server-side annotation costs,
- tuning thresholds to the desired accuracy/latency balance,
- and expanding the gesture vocabulary in a controlled way.

These changes will deliver the best return without requiring major architecture rewrites.
