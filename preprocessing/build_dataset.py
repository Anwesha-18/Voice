"""
GestureSpeak — Build Dataset
build_dataset.py

Loads raw .npy sequences → stratified 80/20 split → saves processed arrays + label map.
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(BASE_DIR, "dataset", "raw_sequences")
PROC_DIR  = os.path.join(BASE_DIR, "dataset", "processed")

WORDS = [
    "hello", "yes", "no", "stop", "thank_you", "help"
]

SEQ_LEN      = 30
FEATURE_SIZE = 126   # 2 hands × 21 landmarks × 3 coords


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_sequences():
    X, y = [], []
    label_map = {}          # int → word
    word_counts = {}

    for idx, word in enumerate(WORDS):
        label_map[idx] = word
        folder = os.path.join(RAW_DIR, word)
        if not os.path.isdir(folder):
            print(f"  [WARN] Missing folder: {folder}")
            word_counts[word] = 0
            continue

        seqs = [f for f in os.listdir(folder) if f.endswith(".npy")]
        word_counts[word] = len(seqs)

        for seq_file in seqs:
            path = os.path.join(folder, seq_file)
            try:
                seq = np.load(path)
                if seq.shape == (SEQ_LEN, FEATURE_SIZE):
                    X.append(seq)
                    y.append(idx)
                else:
                    print(f"  [WARN] Bad shape {seq.shape} in {path} — skipping")
            except Exception as e:
                print(f"  [WARN] Could not load {path}: {e}")

    return np.array(X), np.array(y), label_map, word_counts


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    print("=== GestureSpeak Dataset Builder ===\n")
    print("Loading sequences...")
    X, y, label_map, word_counts = load_sequences()

    print("\nDEBUG:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if len(X) == 0:
        print("\n[ERROR] No sequences found. Run collect_data.py first.")
        return

    print(f"\nSequences loaded per word:")
    for word, count in word_counts.items():
        bar = "█" * count + "░" * max(0, 30 - count)
        print(f"  {word:<12} {bar} {count}")

    print(f"\nTotal: {len(X)} sequences  |  Shape: {X.shape}")
    print(f"Classes: {len(label_map)}")

    # ── Stratified split ──────────────────────
    # Need at least 2 samples per class for stratification
    valid_classes = {cls for cls in np.unique(y) if np.sum(y == cls) >= 2}
    mask = np.array([yi in valid_classes for yi in y])

    if not np.all(mask):
        skipped = len(X) - mask.sum()
        print(f"\n[WARN] Skipping {skipped} samples from classes with <2 sequences.")

    X_valid, y_valid = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, y_valid,
        test_size=0.20,
        random_state=42,
        stratify=y_valid
    )

    # ── Save processed arrays ─────────────────
    np.save(os.path.join(PROC_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROC_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(PROC_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROC_DIR, "y_test.npy"),  y_test)

    # ── Save label map ────────────────────────
    label_map_str = {str(k): v for k, v in label_map.items()}
    with open(os.path.join(PROC_DIR, "label_map.json"), "w") as f:
        json.dump(label_map_str, f, indent=2)

    print(f"\nSplit:")
    print(f"  Train: {len(X_train)} sequences")
    print(f"  Test : {len(X_test)}  sequences")
    print(f"\nSaved to: {PROC_DIR}")
    print("  X_train.npy, X_test.npy, y_train.npy, y_test.npy, label_map.json")


if __name__ == "__main__":
    main()
