"""
VOICE — Real-Time Recognition App  (v3 — final)
app/app.py

Run:  streamlit run app/app.py

Architecture:
  ┌─ UI SECTION ───────────────────────────────────────────────────────┐
  │  All st.button / st.markdown calls live here, ABOVE the loop.     │
  │  Buttons set session_state flags; loop reads + consumes them.     │
  │  This prevents Streamlit reruns from interrupting the webcam loop.│
  └────────────────────────────────────────────────────────────────────┘
  ┌─ WEBCAM LOOP ──────────────────────────────────────────────────────┐
  │  Pure OpenCV + MediaPipe + TF inference.                          │
  │  Never calls st.button() — only st.empty().xxx() for rendering.  │
  │  Sentence append gated by ss.sentence_mode flag.                 │
  └────────────────────────────────────────────────────────────────────┘
  ┌─ TTS WORKER ───────────────────────────────────────────────────────┐
  │  Daemon thread started once at module load.                       │
  │  Main loop puts text on a Queue and immediately continues.       │
  │  Webcam never blocks for speech.                                  │
  └────────────────────────────────────────────────────────────────────┘
"""

import os
import sys
import json
import time
import threading
import collections
import queue as _queue
from queue import Queue
import numpy as np
import cv2
import mediapipe as mp
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VOICE",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "saved_models")
sys.path.insert(0, os.path.join(BASE_DIR, "model"))

# ─────────────────────────────────────────────
# CSS — dark neon theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global reset ── */
  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stApp"] {
    background-color: #0a0a0f !important;
    color: #e0e0e0;
    font-family: 'Courier New', monospace;
  }
  [data-testid="stHeader"],
  [data-testid="stToolbar"],
  footer,
  [data-testid="stSidebar"] { display: none !important; }

  /* ── Title ── */
  .gs-title {
    font-size: 2.8rem; font-weight: 900;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center; letter-spacing: 0.15em; margin-bottom: 0.2rem;
  }
  .gs-subtitle {
    text-align: center; color: #555577;
    font-size: 0.85rem; letter-spacing: 0.3em; margin-bottom: 1.5rem;
  }

  /* ── Detected word ── */
  .gs-word {
    font-size: 4.5rem; font-weight: 900; color: #00d4ff;
    text-align: center;
    text-shadow: 0 0 30px #00d4ff88, 0 0 60px #00d4ff44;
    letter-spacing: 0.08em; line-height: 1.1;
    min-height: 5.5rem;
    display: flex; align-items: center; justify-content: center;
  }
  .gs-idle { color: #333355 !important; text-shadow: none !important; }

  /* ── Confidence bar ── */
  .gs-conf-label {
    color: #888; font-size: 0.75rem;
    letter-spacing: 0.2em; text-align: center; margin-bottom: 0.3rem;
  }
  .gs-conf-bar-bg {
    background: #1a1a2e; border-radius: 6px;
    height: 12px; width: 100%; overflow: hidden;
  }
  .gs-conf-bar-fill {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, #7c3aed, #00d4ff);
    transition: width 0.15s ease;
  }
  .gs-conf-pct {
    color: #00d4ff; font-size: 0.9rem;
    text-align: center; margin-top: 0.25rem; font-weight: 700;
  }

  /* ── Top-3 prediction rows ── */
  .gs-pred-row {
    display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
  }
  .gs-pred-name { width: 90px; font-size: 0.8rem; color: #aaa; text-align: right; }
  .gs-pred-bar-bg {
    flex: 1; background: #1a1a2e; border-radius: 4px;
    height: 8px; overflow: hidden;
  }
  .gs-pred-bar-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #7c3aed88, #00d4ff88);
  }
  .gs-pred-pct { width: 42px; font-size: 0.75rem; color: #666; }

  /* ── Sentence ── */
  .gs-sentence-label {
    color: #444466; font-size: 0.7rem;
    letter-spacing: 0.3em; margin-bottom: 0.4rem;
  }
  .gs-sentence {
    font-size: 1.8rem; color: #ffffff;
    min-height: 3rem; letter-spacing: 0.05em;
    word-break: break-word; line-height: 1.4;
  }
  .gs-sentence-empty { color: #2a2a44; }

  /* ── Mode badges ── */
  .gs-rec-badge {
    display: inline-block;
    background: #200000; border: 1px solid #ff4444;
    color: #ff7777; font-size: 0.7rem; letter-spacing: 0.2em;
    padding: 4px 12px; border-radius: 5px; margin-bottom: 0.6rem;
  }
  .gs-rec-dot {
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; background: #ff4444; margin-right: 5px;
    animation: blink 0.8s infinite;
  }
  .gs-watch-badge {
    display: inline-block;
    background: #000d0a; border: 1px solid #00d4ff33;
    color: #336655; font-size: 0.7rem; letter-spacing: 0.2em;
    padding: 4px 12px; border-radius: 5px; margin-bottom: 0.6rem;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: transparent !important;
    border: 1px solid #333355 !important;
    color: #aaa !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
    padding: 0.35rem 1rem !important;
    transition: all 0.15s !important;
    font-family: 'Courier New', monospace !important;
    width: 100%;
  }
  .stButton > button:hover {
    border-color: #00d4ff !important;
    color: #00d4ff !important;
    box-shadow: 0 0 10px #00d4ff33 !important;
  }

  /* Start / Stop sentence button — first column always gets green */
  div[data-testid="stHorizontalBlock"]
    > div:first-child .stButton > button {
    border-color: #00ff88 !important;
    color: #00ff88 !important;
  }
  div[data-testid="stHorizontalBlock"]
    > div:first-child .stButton > button:hover {
    background: #00ff8811 !important;
    box-shadow: 0 0 10px #00ff8833 !important;
  }

  /* ── Cards ── */
  .gs-card {
    background: #0e0e1a; border: 1px solid #1e1e30;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
  }
  .gs-card-rec {
    background: #0e0e1a; border: 1px solid #ff444422;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
  }

  /* ── Misc ── */
  hr { border-color: #1a1a2e !important; }
  .gs-status { font-size: 0.7rem; color: #444; letter-spacing: 0.2em; text-align: center; }
  .gs-status-live { color: #00d4ff; }
  .gs-status-dot {
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; background: #00d4ff; margin-right: 5px;
    animation: blink 1.2s infinite;
  }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }
  .gs-top-label {
    color: #333355; font-size: 0.7rem;
    letter-spacing: 0.25em; margin-bottom: 0.5rem;
  }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════
SEQ_LEN          = 30
FEATURE_SIZE     = 126
CONF_THRESHOLD   = 0.40   # minimum confidence to display a word
APPEND_THRESHOLD = 0.60   # minimum confidence to append to sentence
APPEND_FRAMES    = 6      # consecutive frames before word is committed (reduced for responsiveness)
SMOOTH_WINDOW    = 3      # majority-vote window (reduced for responsiveness)
IDLE_CLASS       = "idle"

MP_DRAWING  = mp.solutions.drawing_utils
MP_HOLISTIC = mp.solutions.holistic
HAND_STYLE  = MP_DRAWING.DrawingSpec(color=(0, 212, 255), thickness=2, circle_radius=3)
CONN_STYLE  = MP_DRAWING.DrawingSpec(color=(124, 58, 237), thickness=2)


# ═════════════════════════════════════════════
# TTS — one persistent daemon thread + queue
# Started at module import, never restarted.
# Webcam loop calls enqueue_speech() which returns immediately.
# ═════════════════════════════════════════════
_tts_queue: _queue.Queue = _queue.Queue(maxsize=3)

_engine = None

def _tts_worker():
    """Drain the TTS queue in a background thread. Never blocks the main thread."""
    global _engine
    try:
        import pyttsx3
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 150)
    except Exception:
        _engine = None

    while True:
        text = _tts_queue.get()
        if text is None:        # shutdown sentinel
            break
        try:
            if _engine is not None:
                _engine.say(text)
                _engine.runAndWait()
        except Exception:
            pass                # TTS unavailable — fail silently
        finally:
            _tts_queue.task_done()

_tts_thread = threading.Thread(target=_tts_worker, daemon=True, name="gs-tts")
_tts_thread.start()


def enqueue_speech(words):
    """
    Non-blocking TTS trigger.
    Drops silently if the queue already has something pending
    so the webcam loop is NEVER slowed down.
    """
    if isinstance(words, list):
        text = " ".join(words).strip()
    else:
        text = str(words).strip()
    if not text:
        return
    try:
        _tts_queue.put_nowait(text)
    except _queue.Full:
        pass


# ═════════════════════════════════════════════
# MODEL PREDICTION — shared sync runner for Keras/TFLite
# ═════════════════════════════════════════════

class ModelRunner:
    def __init__(self, model, label_map):
        self.model = model
        self.label_map = label_map
        self.use_tflite = hasattr(model, "invoke") and hasattr(model, "get_input_details")
        self.interpreter = model if self.use_tflite else None

    def predict(self, seq: np.ndarray):
        if self.use_tflite and self.interpreter is not None:
            inp_details = self.interpreter.get_input_details()[0]
            out_details = self.interpreter.get_output_details()[0]
            self.interpreter.set_tensor(inp_details['index'], seq.astype(np.float32))
            self.interpreter.invoke()
            probs = self.interpreter.get_tensor(out_details['index'])[0]
        else:
            probs = self.model.predict(seq, verbose=0)[0]

        top3i = np.argsort(probs)[[::-1][:3]
