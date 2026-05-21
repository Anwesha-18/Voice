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
APPEND_FRAMES    = 10     # consecutive frames before word is committed
SMOOTH_WINDOW    = 5      # majority-vote window
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

def _tts_worker():
    """Drain the TTS queue in a background thread. Never blocks the main thread."""
    while True:
        text = _tts_queue.get()
        if text is None:        # shutdown sentinel
            break
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass                # TTS unavailable — fail silently
        finally:
            _tts_queue.task_done()

_tts_thread = threading.Thread(target=_tts_worker, daemon=True, name="gs-tts")
_tts_thread.start()


def enqueue_speech(words: list):
    """
    Non-blocking TTS trigger.
    Drops silently if the queue already has something pending
    so the webcam loop is NEVER slowed down.
    """
    text = " ".join(words).strip()
    if not text:
        return
    try:
        _tts_queue.put_nowait(text)
    except _queue.Full:
        pass


# ═════════════════════════════════════════════
# MODEL LOADING  — cached for the Streamlit session lifetime
# ═════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    model_path  = os.path.join(MODEL_DIR, "best_model.h5")
    labels_path = os.path.join(MODEL_DIR, "label_map.json")

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return None, None

    try:
        import tensorflow as tf
        from architectures import AttentionLayer
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"AttentionLayer": AttentionLayer}
        )
        with open(labels_path) as f:
            raw = json.load(f)
        label_map = {int(k): v for k, v in raw.items()}
        return model, label_map
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None


# ═════════════════════════════════════════════
# SESSION STATE
#
# Three logical groups — none cross-contaminate each other:
#
#   DETECTION    frame_buffer, pred_history, current_word,
#                current_conf, top3
#                → live inference; never reset by UI buttons
#
#   SENTENCE     sentence_mode, sentence, consec_word, consec_count
#                → user-controlled; only modified by explicit actions
#
#   BUTTON FLAGS _do_speak, _do_remove, _do_clear
#                → one-shot; set in UI section, consumed in loop body
#                → prevents Streamlit rerun from interrupting webcam
# ═════════════════════════════════════════════
def init_state():
    defaults = {
        # Detection
        "frame_buffer":  collections.deque(maxlen=SEQ_LEN),
        "pred_history":  collections.deque(maxlen=SMOOTH_WINDOW),
        "current_word":  "",
        "current_conf":  0.0,
        "top3":          [],
        # Sentence
        "sentence_mode": False,
        "sentence":      [],
        "consec_word":   None,
        "consec_count":  0,
        # Button flags
        "_do_speak":     False,
        "_do_remove":    False,
        "_do_clear":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ═════════════════════════════════════════════
# LANDMARK HELPERS — identical to collect_data.py
# DO NOT MODIFY — must stay in sync with training
# ═════════════════════════════════════════════
def normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    wrist = landmarks[0].copy()
    pts   = landmarks - wrist
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def extract_features(results) -> np.ndarray:
    def hand_arr(hl):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark])
        return normalize_hand(pts).flatten()

    left  = hand_arr(results.left_hand_landmarks)  if results.left_hand_landmarks  else np.zeros(63)
    right = hand_arr(results.right_hand_landmarks) if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([left, right])


# ═════════════════════════════════════════════
# INFERENCE HELPERS — model pipeline untouched
# ═════════════════════════════════════════════
def run_prediction(model, label_map, frame_buffer):
    """Returns (word, confidence, top3_list). Returns ("", 0, []) if buffer not full."""
    if len(frame_buffer) < SEQ_LEN:
        return "", 0.0, []
    seq   = np.array(frame_buffer, dtype=np.float32)[np.newaxis]  # (1, 30, 126)
    probs = model.predict(seq, verbose=0)[0]
    top3i = np.argsort(probs)[::-1][:3]
    top3  = [(label_map.get(i, str(i)), float(probs[i])) for i in top3i]
    return top3[0][0], top3[0][1], top3


def smooth_prediction(pred_history, new_word: str) -> str:
    """
    Stable majority-vote smoothing.

    Uses Counter instead of max(set(...)) because:
    - deterministic
    - less flickering
    - stable on ties
    """
    pred_history.append(new_word)

    if len(pred_history) == 0:
        return ""

    return collections.Counter(pred_history).most_common(1)[0][0]


# ═════════════════════════════════════════════
# SENTENCE APPEND — only called when sentence_mode is True
# ═════════════════════════════════════════════
def maybe_append_word(word: str, conf: float) -> bool:
    """
    Commit a word to the sentence when all conditions are met:
      • conf ≥ APPEND_THRESHOLD
      • same word held for APPEND_FRAMES consecutive frames
      • word is not idle
      • word differs from the last committed word
    Returns True if a word was actually appended.
    """
    ss = st.session_state

    # Reset streak on idle / empty / low confidence
    if not word or word == IDLE_CLASS or conf < APPEND_THRESHOLD:
        ss.consec_word  = None
        ss.consec_count = 0
        return False

    # Track consecutive same-word frames
    if word == ss.consec_word:
        ss.consec_count += 1
    else:
        ss.consec_word  = word
        ss.consec_count = 1

    # Commit when streak threshold reached
    if ss.consec_count >= APPEND_FRAMES:
        if not ss.sentence or ss.sentence[-1] != word:
            ss.sentence     = ss.sentence + [word]
            ss.consec_count = 0   # reset streak so same word can be added again later
            return True

    return False


# ═════════════════════════════════════════════
# DRAW ON FRAME
# ═════════════════════════════════════════════
def draw_overlay(frame, results, word: str, conf: float, sentence_mode: bool):
    h, w = frame.shape[:2]

    # Hand skeleton
    if results.left_hand_landmarks:
        MP_DRAWING.draw_landmarks(
            frame, results.left_hand_landmarks,
            MP_HOLISTIC.HAND_CONNECTIONS, HAND_STYLE, CONN_STYLE
        )
    if results.right_hand_landmarks:
        MP_DRAWING.draw_landmarks(
            frame, results.right_hand_landmarks,
            MP_HOLISTIC.HAND_CONNECTIONS, HAND_STYLE, CONN_STYLE
        )

    # Top-right LIVE indicator
    cv2.circle(frame, (w - 18, 18), 7, (0, 212, 255), -1)
    cv2.putText(frame, "LIVE", (w - 50, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 212, 255), 1)

    # REC badge when sentence mode is active
    if sentence_mode:
        cv2.rectangle(frame, (w - 80, 30), (w - 10, 52), (50, 0, 0), -1)
        cv2.rectangle(frame, (w - 80, 30), (w - 10, 52), (200, 0, 0), 1)
        cv2.circle(frame,   (w - 70, 41), 5, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 60, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 255), 1)

    # Bottom: rolling buffer fill bar (shows how full the 30-frame window is)
    buf_len = len(st.session_state.frame_buffer)
    bar_w   = int((buf_len / SEQ_LEN) * (w - 20))
    cv2.rectangle(frame, (10, h - 10), (w - 10, h - 5), (20, 20, 35), -1)
    cv2.rectangle(frame, (10, h - 10), (10 + bar_w, h - 5), (124, 58, 237), -1)

    return frame


# ═════════════════════════════════════════════
# HTML HELPERS
# ═════════════════════════════════════════════
def conf_bar_html(label: str, conf: float, primary=False) -> str:
    pct = conf * 100
    if primary:
        return (
            '<div class="gs-conf-label">CONFIDENCE</div>'
            '<div class="gs-conf-bar-bg">'
            f'  <div class="gs-conf-bar-fill" style="width:{pct:.1f}%"></div>'
            '</div>'
            f'<div class="gs-conf-pct">{pct:.1f}%</div>'
        )
    return (
        '<div class="gs-pred-row">'
        f'  <div class="gs-pred-name">{label}</div>'
        '  <div class="gs-pred-bar-bg">'
        f'    <div class="gs-pred-bar-fill" style="width:{pct:.1f}%"></div>'
        '  </div>'
        f'  <div class="gs-pred-pct">{pct:.1f}%</div>'
        '</div>'
    )


def mode_badge_html(active: bool) -> str:
    if active:
        return (
            '<div class="gs-rec-badge">'
            '<span class="gs-rec-dot"></span>SENTENCE MODE ON'
            '</div>'
        )
    return '<div class="gs-watch-badge">DETECTION ONLY</div>'


# ═════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════
def main():
    ss = st.session_state

    # ── Header ─────────────────────────────────────────────────────
    st.markdown('<div class="gs-title">🤟 VOICE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gs-subtitle">REAL-TIME SIGN LANGUAGE RECOGNITION</div>',
        unsafe_allow_html=True
    )

    # ── Model load ─────────────────────────────────────────────────
    model, label_map = load_model_and_labels()
    if model is None:
        st.markdown("""
<div style="text-align:center;padding:3rem;color:#555577;">
  <div style="font-size:3rem;margin-bottom:1rem;">⚙️</div>
  <div style="color:#00d4ff;font-size:1.1rem;margin-bottom:1rem;">No trained model found.</div>
  <div style="font-size:0.85rem;line-height:2.2;">
    1. <code>python preprocessing/collect_data.py</code><br>
    2. <code>python preprocessing/build_dataset.py</code><br>
    3. <code>python model/train.py</code><br>
    4. <code>streamlit run app/app.py</code>
  </div>
</div>""", unsafe_allow_html=True)
        return

    # ══════════════════════════════════════════════════════════════
    # UI SECTION — ALL buttons declared here, above the webcam loop
    #
    # Why: Streamlit re-runs the whole script on every button click.
    # If buttons were inside the webcam loop, each click would
    # reset the cap / holistic objects. Instead:
    #   • buttons set ss flags  (this section)
    #   • loop reads + clears flags  (loop section below)
    #   • webcam + MediaPipe objects are created ONCE, in the loop
    # ══════════════════════════════════════════════════════════════

    col_cam, col_right = st.columns([3, 2], gap="large")

    with col_cam:
        cam_placeholder    = st.empty()
        status_placeholder = st.empty()

    with col_right:
        mode_badge_ph  = st.empty()
        word_ph        = st.empty()
        conf_ph        = st.empty()
        st.markdown('<div class="gs-top-label">TOP PREDICTIONS</div>', unsafe_allow_html=True)
        top3_ph        = st.empty()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Sentence display area ──────────────────
    sent_badge_ph = st.empty()
    sent_ph       = st.empty()

    # ── Controls row ───────────────────────────
    # Layout: [Start/Stop Sentence] [▶ Speak] [⌫ Remove] [✕ Clear]
    c1, c2, c3, c4, _ = st.columns([1.4, 1, 1, 1, 1.6])

    with c1:
        toggle_label = "⏹ Stop Sentence" if ss.sentence_mode else "● Start Sentence"
        if st.button(toggle_label, key="btn_toggle"):
            ss.sentence_mode = not ss.sentence_mode
            ss.consec_word   = None   # reset streak on mode change
            ss.consec_count  = 0

    with c2:
        if st.button("▶  Speak", key="btn_speak"):
            ss._do_speak = True       # flag consumed below

    with c3:
        if st.button("⌫  Remove", key="btn_remove"):
            ss._do_remove = True

    with c4:
        if st.button("✕  Clear", key="btn_clear"):
            ss._do_clear = True

    # Stop camera sits on its own row so it doesn't share a column group
    stop_ph  = st.empty()
    stop_btn = stop_ph.button("⏹  Stop Camera", key="btn_stop")

    # ── Consume button flags immediately ───────
    # These are processed here (before the loop) so the action takes
    # effect even if the rerun fires before the next loop iteration.
    if ss._do_speak:
        enqueue_speech(ss.sentence)   # non-blocking — TTS daemon handles it
        ss._do_speak = False

    if ss._do_remove:
        if ss.sentence:
            ss.sentence = ss.sentence[:-1]
        ss._do_remove = False

    if ss._do_clear:
        ss.sentence     = []
        ss.consec_word  = None
        ss.consec_count = 0
        ss._do_clear    = False

    # ══════════════════════════════════════════════════════════════
    # WEBCAM + INFERENCE LOOP
    #
    # cap and holistic are created once per page load.
    # They are NOT recreated on button clicks because buttons only
    # trigger Streamlit reruns that restart main(); cap/holistic are
    # local to this block, so they do get recreated per rerun —
    # but that is the minimum necessary and keeps the code simple.
    # For a truly persistent camera across reruns, a threading model
    # would be needed; this approach gives good stability for most use.
    # ══════════════════════════════════════════════════════════════
    holistic = MP_HOLISTIC.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,
    )
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    try:
        while not stop_btn:
            # ── Read frame ────────────────────────
            ret, frame = cap.read()
            if not ret:
                status_placeholder.markdown(
                    '<div class="gs-status">⚠  Camera unavailable — retrying…</div>',
                    unsafe_allow_html=True
                )
                time.sleep(0.05)
                continue

            # ── MediaPipe inference ───────────────
            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # ── Feature extraction + buffer ───────
            feats = extract_features(results)
            ss.frame_buffer.append(feats)

            # ── Model prediction ──────────────────
            word, conf, top3 = run_prediction(
                model,
                label_map,
                ss.frame_buffer
            )

            # Only smooth HIGH-confidence predictions.
            # Low-confidence predictions pollute history and create flicker.
            if conf >= CONF_THRESHOLD:

                smoothed = smooth_prediction(
                    ss.pred_history,
                    word
                )

                display_word = smoothed
                display_conf = conf

            else:
                # Clear stale history so old predictions
                # don't linger after confidence drops.
                ss.pred_history.clear()

                smoothed = ""
                display_word = ""
                display_conf = 0.0

            ss.current_word = display_word
            ss.current_conf = display_conf
            ss.top3         = top3

            # ── Sentence append (ONLY when mode is active) ────
            if ss.sentence_mode:
                maybe_append_word(smoothed, conf)

            # ── Render webcam frame ───────────────
            frame_rgb = draw_overlay(
                rgb.copy(), results, display_word, display_conf, ss.sentence_mode
            )
            cam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            status_placeholder.markdown(
                '<div class="gs-status gs-status-live">'
                '<span class="gs-status-dot"></span>LIVE</div>',
                unsafe_allow_html=True
            )

            # ── Mode badge ────────────────────────
            mode_badge_ph.markdown(mode_badge_html(ss.sentence_mode), unsafe_allow_html=True)

            # ── Detected word ─────────────────────
            is_idle    = (not display_word) or (display_word == IDLE_CLASS)
            word_cls   = "gs-idle" if is_idle else ""
            show_label = "···" if is_idle else display_word.upper().replace("_", " ")
            word_ph.markdown(
                f'<div class="gs-word {word_cls}">{show_label}</div>',
                unsafe_allow_html=True
            )

            # ── Confidence bar ────────────────────
            card_cls = "gs-card-rec" if ss.sentence_mode else "gs-card"
            conf_ph.markdown(
                f'<div class="{card_cls}">'
                f'{conf_bar_html("", display_conf, primary=True)}'
                f'</div>',
                unsafe_allow_html=True
            )

            # ── Top-3 ─────────────────────────────
            if top3:
                t3_html = "".join(
                    conf_bar_html(w.replace("_", " "), c)
                    for w, c in top3
                )
                top3_ph.markdown(
                    f'<div class="gs-card">{t3_html}</div>',
                    unsafe_allow_html=True
                )

            # ── Sentence display ──────────────────
            sent_badge_ph.markdown(mode_badge_html(ss.sentence_mode), unsafe_allow_html=True)

            if ss.sentence:
                sent_text = " · ".join(w.replace("_", " ").upper() for w in ss.sentence)
                sent_html = f'<div class="gs-sentence">{sent_text}</div>'
            elif ss.sentence_mode:
                sent_html = (
                    '<div class="gs-sentence gs-sentence-empty">'
                    '[ hold a sign for ~10 frames to add it ]'
                    '</div>'
                )
            else:
                sent_html = (
                    '<div class="gs-sentence gs-sentence-empty">'
                    '[ press  ● Start Sentence  to begin building ]'
                    '</div>'
                )

            sent_ph.markdown(
                f'<div class="gs-sentence-label">SENTENCE</div>{sent_html}',
                unsafe_allow_html=True
            )

    finally:
        cap.release()
        holistic.close()
        status_placeholder.markdown(
            '<div class="gs-status">Camera stopped.</div>',
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()