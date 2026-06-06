# VOICE

**VOICE** is a real-time sign language communication system built with:
- a browser frontend using webcam capture,
- a Flask backend with MediaPipe hand landmark extraction,
- a BiLSTM + attention classifier,
- sentence building and speech output.

This project is designed to convert short hand gesture sequences into words, then build simple phrases for assisted communication.

---

## Project overview

The repository is organized into four main layers:

- `dataset/` — raw `.npy` gesture sequences and processed training arrays.
- `preprocessing/` — data collection and dataset construction scripts.
- `model/` — deep learning architecture, training logic, and TFLite conversion support.
- `backend/` — Flask inference server, prediction endpoints, and optional Gemini sentence generation.
- `frontend/` — React + Vite app for live webcam capture, prediction display, and speech output.

---

## Supported workflow

1. Collect gesture data with `preprocessing/collect_data.py`
2. Build the processed dataset with `preprocessing/build_dataset.py`
3. Train the sequence model with `model/train.py`
4. Run the backend and frontend together for live recognition

---

## Installation

### Python backend

1. Create and activate a Python virtual environment.
2. Install backend dependencies:

```bash
cd backend
pip install -r requirements.txt
```

If you are using Apple Silicon, install `tensorflow-macos` instead of the default `tensorflow` package.

### Frontend

1. Install Node dependencies:

```bash
cd frontend
npm install
```

2. Start the app:

```bash
npm run dev
```

The frontend runs at `http://localhost:3000` and communicates with the Flask backend on `http://localhost:5000`.

---

## Data collection

The collection script records 30-frame sequences of hand landmarks and saves them as `.npy` files.

### Current words tracked by `collect_data.py`

- `hello`
- `yes`
- `no`
- `stop`
- `thank_you`
- `help`
- `food`
- `water`
- 'medicine'
- 'doctor'
- 'please'
- 'bathroom'

> Note: The repository contains additional raw folders, but the current collection and dataset builder scripts are configured for these 8 words.

### Controls

- `SPACE` — start recording a sequence after a 3-second countdown
- `N` — advance to the next word
- `P` — go back to the previous word
- `Q` — quit the collector

### Recording details

- Each sequence uses `SEQ_LEN = 30` frames.
- Each frame encodes `FEATURE_SIZE = 126` features.
- Default target sequences per word: `TARGET_SEQS = 50`.
- The script auto-advances to the next word after hitting the target count.

### Recommended capture tips

- Keep lighting even and avoid harsh shadows.
- Move hands naturally, but keep each sign clearly separated.
- Capture both left and right hand poses when available.
- Save multiple examples of the same word to improve generalization.

---

## Building the dataset

Run:

```bash
python preprocessing/build_dataset.py
```

What it does:

- loads raw `.npy` sequences from `dataset/raw_sequences`
- validates each sequence shape is `(30, 126)`
- performs a stratified `80/20` train/test split
- saves:
  - `dataset/processed/X_train.npy`
  - `dataset/processed/X_test.npy`
  - `dataset/processed/y_train.npy`
  - `dataset/processed/y_test.npy`
  - `dataset/processed/label_map.json`

The dataset builder currently requires at least 2 samples per class for stratified splitting.

---

## Training the model

Run:

```bash
python model/train.py
```

Training pipeline details:

- Uses a BiLSTM + attention architecture from `model/architectures.py`.
- Applies Gaussian noise augmentation to training sequences.
- Uses class-weight balancing for imbalanced gesture labels.
- Monitors validation accuracy and saves the best checkpoint to `outputs/saved_models/best_model.h5`.
- Generates:
  - `outputs/training_log.csv`
  - `outputs/training_curves.png`
  - `outputs/confusion_matrix.png`
- Also saves the label map and model metadata in `outputs/saved_models/`.

### Training parameters

- `SEQ_LEN = 30`
- `FEATURE_SIZE = 126`
- `EPOCHS = 50`
- `BATCH_SIZE = 32`
- `NOISE_STD = 0.005`

---

## Model architecture

The model accepts a `30 × 126` input sequence and computes a class probability over gesture labels.

Architecture summary:

- Input layer: `(30, 126)`
- Bidirectional LSTM 128 → dropout 0.2 + 0.3
- Bidirectional LSTM 64 → dropout 0.2 + 0.3
- Custom attention layer over timesteps
- Dense 128 → BatchNorm → dropout 0.4
- Dense 64 → dropout 0.3
- Output dense with softmax over classes

### Feature representation

- Each frame contains 126 features:
  - left hand: 21 landmarks × 3 coordinates
  - right hand: 21 landmarks × 3 coordinates
- Each hand is wrist-normalized and scale-normalized.
- Missing hands are represented by zero vectors.
- Face landmarks are excluded to keep the model small and privacy-friendly.

---

## Backend inference server

Run the backend with:

```bash
python backend/app.py
```

The server provides:

- `POST /api/predict` — accepts base64 JPEG webcam frames and returns predictions
- `POST /api/generate-sentence` — converts collected words into a natural sentence
- `GET /api/metrics` — returns backend health and buffer statistics
- `GET /api/ping` — simple health check

### Backend behavior

- Loads `outputs/saved_models/best_model.tflite` if available.
- Falls back to `best_model.h5` otherwise.
- Uses per-client MediaPipe Holistic state for stable hand landmark tracking.
- Runs MediaPipe every `MP_FRAME_SKIP = 3` frames to improve latency.
- Uses `CONF_THRESHOLD = 0.55` to suppress low-confidence predictions.
- Only non-`idle` words pass to the frontend.

### Optional sentence generation

The sentence generation endpoint tries to call Gemini if `GEMINI_API_KEY` is set.
If Gemini is unavailable, the backend returns a heuristic fallback sentence.

To use Gemini, add a `.env` file in the repo root with:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## Frontend app

The frontend is implemented in React with Vite and renders a neon-style live camera interface.

Key features:

- Captures video from the browser webcam.
- Downscales frames to `320×240` before sending to the backend.
- Sends JPEG images at quality `0.30` for smaller payloads.
- Displays live hand landmark overlays.
- Shows current prediction, confidence, and buffer progress.
- Automatically speaks recognized words using browser TTS.
- Supports assisted communication mode with sentence generation.

### Frontend thresholds

The current frontend thresholds are:

- `CAPTURE_INTERVAL_MS = 25` — captures frames at up to 40 FPS
- `APPEND_THRESHOLD = 0.50` — confidence threshold for word appending
- `APPEND_FRAMES = 1` — appends a word after one confident prediction
- Back-to-back word cooldown: `600 ms`

---

## Running the full system

1. Start the backend:

```bash
cd backend
python app.py
```

2. Start the frontend:

```bash
cd frontend
npm run dev
```

3. Open `http://localhost:3000`

4. Allow camera access when prompted.

---

## Troubleshooting

### Common issues

- `Model not loaded`
  - Ensure `outputs/saved_models/best_model.h5` or `best_model.tflite` exists.
  - Run `python model/train.py` if no model is present.

- `Unable to access camera`
  - Check browser permissions.
  - Confirm the correct webcam is selected.

- `Gemini API key missing`
  - Add `GEMINI_API_KEY` to `.env` if you want sentence generation.

- Slow performance
  - Confirm backend uses TFLite if available.
  - Reduce frame rate or lower model complexity.

### Notes

- `dataset/raw_sequences/` may contain more folders than the current dataset builder uses.
- To add new gesture classes, update the `WORDS` list in `preprocessing/collect_data.py` and `preprocessing/build_dataset.py`.

---

## Directory reference

```text
VOICE/
├── backend/                # Flask inference server + dependencies
├── dataset/
│   ├── processed/          # saved training arrays and label map
│   └── raw_sequences/      # collected gesture `.npy` sequences
├── frontend/               # React UI and live webcam experience
├── model/                  # model architecture and training logic
├── outputs/                # training outputs and saved model artifacts
├── preprocessing/          # data collection and dataset builder scripts
├── OPTIMIZATION_REPORT.md  # latency and performance analysis
└── README.md               # this file
```

---

## License

MIT License.
