"""
GestureSpeak — TFLite Converter
convert_tflite.py

Run ONCE after training:
    python convert_tflite.py

What it does:
  1. Loads your trained best_model.h5
  2. Converts it to TFLite format (same weights, no retraining)
  3. Saves best_model.tflite next to best_model.h5

Why:
  TFLite runs 5-10x faster than full Keras for single-sample
  real-time inference. Accuracy is identical.
"""

import os
import sys

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname (os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "outputs", "saved_models")
H5_PATH    = os.path.join(MODEL_DIR, "best_model.h5")
TFLITE_PATH = os.path.join(MODEL_DIR, "best_model.tflite")

sys.path.insert(0, os.path.join(BASE_DIR, "model"))

# ── Check source exists ──────────────────────────────────────────────
if not os.path.exists(H5_PATH):
    print(f"\n[ERROR] Could not find: {H5_PATH}")
    print("  Run model/train.py first to produce best_model.h5")
    sys.exit(1)

print("=== GestureSpeak TFLite Converter ===\n")
print(f"  Source : {H5_PATH}")
print(f"  Output : {TFLITE_PATH}\n")

# ── Load model ───────────────────────────────────────────────────────
print("Loading model...")
import tensorflow as tf
from architectures import AttentionLayer

model = tf.keras.models.load_model(
    H5_PATH,
    custom_objects={"AttentionLayer": AttentionLayer}
)
print(f"  Input  shape : {model.input_shape}")
print(f"  Output shape : {model.output_shape}")
print(f"  Parameters   : {model.count_params():,}")

# ── Convert ──────────────────────────────────────────────────────────
print("\nConverting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# BiLSTM uses TensorList ops which standard TFLite can't handle.
# SELECT_TF_OPS tells the converter to fall back to full TF kernels
# for those ops while keeping everything else as native TFLite.
# Accuracy is identical — this is purely a compatibility setting.
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

# ── Save ─────────────────────────────────────────────────────────────
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

h5_size     = os.path.getsize(H5_PATH)     / (1024 * 1024)
tflite_size = os.path.getsize(TFLITE_PATH) / (1024 * 1024)

print(f"\n  Done.")
print(f"  best_model.h5     : {h5_size:.1f} MB")
print(f"  best_model.tflite : {tflite_size:.1f} MB")
print(f"\n✓ Saved to: {TFLITE_PATH}")
print("\nYou can now run:  streamlit run app.py\n")