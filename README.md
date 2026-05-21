# 🤟 GestureSpeak

**Real-time sign language recognition system** — recognises 21 ASL-inspired signs from your webcam, builds sentences, and speaks them aloud.

---

## Quick Start

```
GESTURESPEAK/
├── dataset/
│   └── raw_sequences/<word>/seq_XXX.npy
├── preprocessing/
│   ├── collect_data.py      ← Step 1
│   └── build_dataset.py     ← Step 2
├── model/
│   ├── architectures.py
│   └── train.py             ← Step 3
├── app/
│   └── app.py               ← Step 4 (Streamlit UI)
├── outputs/
│   └── saved_models/
│       ├── best_model.h5
│       ├── label_map.json
│       └── model_config.json
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

> Python 3.9–3.11 recommended. On Apple Silicon, install `tensorflow-macos` instead of `tensorflow`.

---

## Step-by-Step Usage

### 1 — Collect Data

```bash
cd GESTURESPEAK
python preprocessing/collect_data.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `SPACE` | Start recording a sequence (3-second countdown) |
| `N` | Next word |
| `P` | Previous word |
| `Q` | Quit |

- Records **30 sequences × 21 words = 630 sequences** total
- Each sequence = 30 frames of hand landmarks
- Auto-advances when a word reaches its target (30 sequences)
- Progress bar shows how many sequences have been captured

**Tips for good data:**
- Record `idle` with hands resting still, fidgeting, and mid-transition between signs
- Vary lighting, hand position slightly, and speed across sequences
- `idle` is the most important class — it prevents false positives when you're not signing

---

### 2 — Build Dataset

```bash
python preprocessing/build_dataset.py
```

- Loads all `.npy` sequences
- Stratified 80/20 train/test split
- Saves `dataset/processed/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`
- Saves `label_map.json`

---

### 3 — Train

```bash
python model/train.py
```

- Trains **BiLSTM + Attention** model for up to 50 epochs
- EarlyStopping (patience=10), ReduceLROnPlateau, ModelCheckpoint
- Best model saved to `outputs/saved_models/best_model.h5`
- Accuracy/loss curves → `outputs/training_curves.png`
- Confusion matrix → `outputs/confusion_matrix.png`
- Training log → `outputs/training_log.csv`

Typical training time: 5–15 min on CPU, 1–3 min with GPU.

---

### 4 — Run App

```bash
streamlit run app/app.py
```

Opens in your browser at `http://localhost:8501`.

**Features:**
- 🎥 Live webcam with hand skeleton overlay
- 🔤 Large neon word display with confidence score
- 📊 Top-3 prediction confidence bars
- 📝 Sentence builder (auto-appends confirmed words)
- 🔊 **Speak** button reads your sentence aloud
- ⌫ Remove last word / ✕ Clear sentence

**Word is appended to sentence when:**
- Confidence > 60%
- Same word held for 10+ consecutive frames
- Word is not `idle`
- Word differs from the last appended word

---

## Recognised Signs (21 classes)

`hello` · `yes` · `no` · `please` · `thank_you` · `sorry` · `help` · `stop` · `eat` · `food` · `drink` · `water` · `who` · `where` · `what` · `i` · `want` · `need` · `bathroom` · `doctor` · `idle`

---

## Model Architecture

```
Input (30, 126)
  → BiLSTM(128, return_sequences=True) + Dropout(0.3)
  → BiLSTM(64,  return_sequences=True) + Dropout(0.3)
  → Soft Attention (weighted sum over timesteps)
  → Dense(128, relu) → BatchNorm → Dropout(0.4)
  → Dense(64, relu)  → Dropout(0.3)
  → Dense(21, softmax)
```

**Features:** 126 per frame = left hand (63) + right hand (63)  
Each hand = 21 landmarks × 3 (x, y, z), wrist-normalised and scale-normalised.  
Face landmarks deliberately excluded to keep the model lightweight and privacy-friendly.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Webcam not detected | Try `cv2.VideoCapture(1)` or `(2)` |
| Model not loading | Ensure step 1–3 are complete; check `outputs/saved_models/` |
| TTS silent | `pyttsx3` may need system TTS engine; try `espeak` on Linux |
| Low accuracy | Collect more `idle` data; ensure consistent lighting |
| `idle` triggers mid-sign | Increase `APPEND_FRAMES` constant in `app.py` |

---

## License

MIT — use freely for personal, educational, and research purposes.
