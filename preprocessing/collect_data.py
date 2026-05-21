"""
GestureSpeak — Data Collection
collect_data.py

Controls:
  SPACE  → start recording a sequence
  N      → next word
  P      → previous word
  Q      → quit
"""

import os
import time
import numpy as np
import cv2
import mediapipe as mp

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WORDS = [
    "hello", "yes", "no", "stop", "thank_you", "help"
    ]

SEQ_LEN        = 30          # frames per sequence
TARGET_SEQS    = 30          # sequences per word
COUNTDOWN_SECS = 3
DATA_ROOT      = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw_sequences")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def ensure_dirs():
    for word in WORDS:
        os.makedirs(os.path.join(DATA_ROOT, word), exist_ok=True)


def count_existing(word: str) -> int:
    folder = os.path.join(DATA_ROOT, word)
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".npy")])


def next_seq_index(word: str) -> int:
    folder = os.path.join(DATA_ROOT, word)
    existing = [f for f in os.listdir(folder) if f.endswith(".npy")]
    if not existing:
        return 0
    indices = [int(f.split("_")[1].split(".")[0]) for f in existing]
    return max(indices) + 1


def normalize_hand(landmarks):
    """
    landmarks: (21, 3) array
    Subtract wrist (index 0), then scale by max distance from wrist.
    Returns (21, 3) normalized.
    """
    wrist = landmarks[0].copy()
    pts   = landmarks - wrist
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def extract_features(results) -> np.ndarray:
    """
    Returns (126,) = left_hand (63,) + right_hand (63,).
    Missing hand → zeros.
    """
    def hand_arr(hand_landmarks):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        return normalize_hand(pts).flatten()

    left  = hand_arr(results.left_hand_landmarks)  if results.left_hand_landmarks  else np.zeros(63)
    right = hand_arr(results.right_hand_landmarks) if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([left, right])


# ─────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────
MP_DRAWING  = mp.solutions.drawing_utils
MP_HOLISTIC = mp.solutions.holistic
HAND_STYLE  = MP_DRAWING.DrawingSpec(color=(0, 212, 255), thickness=2, circle_radius=3)
CONN_STYLE  = MP_DRAWING.DrawingSpec(color=(124, 58, 237), thickness=2)


def draw_landmarks(frame, results):
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


def draw_ui(frame, word_idx, word, collected, is_recording, frame_idx, countdown):
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Word name
    cv2.putText(frame, f"Word [{word_idx+1}/{len(WORDS)}]: {word.upper()}",
                (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 212, 255), 2)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 12, 50, 300, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 60), -1)
    fill = int(bar_w * min(collected / TARGET_SEQS, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (124, 58, 237), -1)
    cv2.putText(frame, f"{collected}/{TARGET_SEQS}", (bar_x + bar_w + 8, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Status
    if countdown > 0:
        txt = f"Starting in {countdown}..."
        color = (0, 212, 255)
    elif is_recording:
        txt = f"Recording  {frame_idx}/{SEQ_LEN}"
        color = (0, 80, 255)
        # Red dot
        cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
    else:
        txt = "SPACE=record  N=next  P=prev  Q=quit"
        color = (180, 180, 180)

    cv2.putText(frame, txt, (12, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Done banner
    if collected >= TARGET_SEQS:
        cv2.putText(frame, "COMPLETE!", (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 100), 3)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    ensure_dirs()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    word_idx     = 0
    is_recording = False
    frame_idx    = 0
    buffer       = []
    countdown    = 0
    cd_start     = 0

    holistic = MP_HOLISTIC.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("\n=== GestureSpeak Data Collector ===")
    print("SPACE=record  N=next  P=prev  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        draw_landmarks(frame, results)

        word      = WORDS[word_idx]
        collected = count_existing(word)

        # ── Countdown phase ──────────────────────────
        if countdown > 0:
            elapsed  = time.time() - cd_start
            remaining = COUNTDOWN_SECS - int(elapsed)
            if remaining <= 0:
                countdown    = 0
                is_recording = True
                frame_idx    = 0
                buffer       = []
            else:
                countdown = remaining

        # ── Recording phase ──────────────────────────
        if is_recording:
            feats = extract_features(results)
            buffer.append(feats)
            frame_idx += 1

            if frame_idx >= SEQ_LEN:
                seq      = np.array(buffer)           # (30, 126)
                idx_file = next_seq_index(word)
                path     = os.path.join(DATA_ROOT, word, f"seq_{idx_file:03d}.npy")
                np.save(path, seq)
                print(f"  Saved {word}/seq_{idx_file:03d}.npy  [{collected+1}/{TARGET_SEQS}]")
                is_recording = False
                frame_idx    = 0
                buffer       = []

                # Auto-advance
                if collected + 1 >= TARGET_SEQS:
                    print(f"  ✓ {word} complete — advancing to next word")
                    time.sleep(0.5)
                    word_idx = (word_idx + 1) % len(WORDS)

        draw_ui(frame, word_idx, word, collected, is_recording, frame_idx, countdown)
        cv2.imshow("GestureSpeak — Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and not is_recording and countdown == 0:
            if collected < TARGET_SEQS:
                countdown = COUNTDOWN_SECS
                cd_start  = time.time()
            else:
                print(f"  {word} already has {collected} sequences. Press N to advance.")
        elif key == ord('n'):
            word_idx = (word_idx + 1) % len(WORDS)
            is_recording = False
        elif key == ord('p'):
            word_idx = (word_idx - 1) % len(WORDS)
            is_recording = False

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()

    # ── Summary ──────────────────────────────
    print("\n=== Collection Summary ===")
    total = 0
    for w in WORDS:
        n = count_existing(w)
        total += n
        status = "✓" if n >= TARGET_SEQS else f"{n}/{TARGET_SEQS}"
        print(f"  {w:<12} {status}")
    print(f"\nTotal sequences: {total}")


if __name__ == "__main__":
    main()
