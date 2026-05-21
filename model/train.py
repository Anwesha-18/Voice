"""
GestureSpeak — Training
train.py

Loads processed dataset → trains BiLSTM+Attention → saves best model.
"""

import os
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

# Local import
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.architectures import build_bilstm_attention

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "dataset", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "saved_models")

SEQ_LEN      = 30
FEATURE_SIZE = 126
EPOCHS       = 50
BATCH_SIZE   = 32
NOISE_STD    = 0.005   # Gaussian noise augmentation


# ─────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────
def augment_sequences(X: np.ndarray, std: float = NOISE_STD) -> np.ndarray:
    """Add small Gaussian noise to training sequences."""
    noise = np.random.normal(0, std, X.shape).astype(np.float32)
    return X + noise


# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────
def plot_history(history, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0a0f")

    for ax in (ax1, ax2):
        ax.set_facecolor("#12121c")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("#00d4ff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    epochs = range(1, len(history.history["accuracy"]) + 1)

    # Accuracy
    ax1.plot(epochs, history.history["accuracy"],     color="#00d4ff", lw=2, label="Train")
    ax1.plot(epochs, history.history["val_accuracy"], color="#7c3aed", lw=2, linestyle="--", label="Val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white")

    # Loss
    ax2.plot(epochs, history.history["loss"],     color="#00d4ff", lw=2, label="Train")
    ax2.plot(epochs, history.history["val_loss"], color="#7c3aed", lw=2, linestyle="--", label="Val")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, label_map, save_dir):
    labels = [label_map[str(i)] for i in sorted(int(k) for k in label_map)]
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="viridis",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5, linecolor="#333355",
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Normalised Confusion Matrix", color="#00d4ff", fontsize=14)
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("True", color="white")
    ax.tick_params(colors="white", labelsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== GestureSpeak Trainer ===\n")

    # ── Load data ────────────────────────────
    print("Loading dataset...")
    X_train = np.load(os.path.join(PROC_DIR, "X_train.npy")).astype(np.float32)
    X_test  = np.load(os.path.join(PROC_DIR, "X_test.npy")).astype(np.float32)
    y_train = np.load(os.path.join(PROC_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(PROC_DIR, "y_test.npy"))

    with open(os.path.join(PROC_DIR, "label_map.json")) as f:
        label_map = json.load(f)   # str(int) → word

    num_classes = len(label_map)
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}  |  Classes: {num_classes}")

    # ── Augment training data ─────────────────
    X_train_aug = augment_sequences(X_train)
    X_train_all = np.concatenate([X_train, X_train_aug], axis=0)
    y_train_all = np.concatenate([y_train, y_train], axis=0)
    print(f"  After augmentation — Train: {len(X_train_all)}")

    # ── One-hot encode ────────────────────────
    Y_train = to_categorical(y_train_all, num_classes)
    Y_test  = to_categorical(y_test, num_classes)

    # ── Class weights ─────────────────────────
    cw = compute_class_weight("balanced", classes=np.unique(y_train_all), y=y_train_all)
    class_weight_dict = {i: w for i, w in enumerate(cw)}

    # ── Build model ───────────────────────────
    print("\nBuilding model...")
    model = build_bilstm_attention(SEQ_LEN, FEATURE_SIZE, num_classes)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # ── Callbacks ─────────────────────────────
    best_model_path = os.path.join(MODEL_DIR, "best_model.h5")
    callbacks = [
        ModelCheckpoint(best_model_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1),
        CSVLogger(os.path.join(OUTPUT_DIR, "training_log.csv")),
    ]

    # ── Train ─────────────────────────────────
    print("\nTraining...")
    history = model.fit(
        X_train_all, Y_train,
        validation_data=(X_test, Y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # ── Save artifacts ────────────────────────
    shutil.copy(os.path.join(PROC_DIR, "label_map.json"),
                os.path.join(MODEL_DIR, "label_map.json"))

    model_config = {"seq_len": SEQ_LEN, "feature_size": FEATURE_SIZE, "num_classes": num_classes}
    with open(os.path.join(MODEL_DIR, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    # ── Evaluate ──────────────────────────────
    print("\n=== Evaluation ===")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    target_names = [label_map[str(i)] for i in range(num_classes)]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    print("\nPer-class accuracy:")
    for i, name in enumerate(target_names):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"  {name:<12} {bar} {acc:.2%}")

    # ── Plots ─────────────────────────────────
    print("\nSaving plots...")
    plot_history(history, OUTPUT_DIR)
    plot_confusion_matrix(y_test, y_pred, label_map, OUTPUT_DIR)

    print(f"\n✓ Best model → {best_model_path}")
    print("Done.\n")


if __name__ == "__main__":
    main()
