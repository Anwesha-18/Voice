import base64
import json
import os
import sys
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# Ensure the root repo path is available for importing model.architectures
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from model.architectures import AttentionLayer

# Flask app
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

# Model / prediction settings
SEQ_LEN = 30
FEATURE_SIZE = 126
CONF_THRESHOLD = 0.70  # Increased: only predict when very confident (was 0.40)
APPEND_THRESHOLD = 0.60
CLIENT_TIMEOUT_SECONDS = 90

MODEL_DIR = os.path.join(ROOT_DIR, "outputs", "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
TFLITE_PATH = os.path.join(MODEL_DIR, "best_model.tflite")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

# Global inference state
model_lock = threading.Lock()
model_runner = None
label_map = {}
client_buffers = {}
client_lock = threading.Lock()

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0,  # Lightweight model for faster two-hand detection
)


def decode_image_data(data_url: str):
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    data = base64.b64decode(encoded)
    image = np.frombuffer(data, dtype=np.uint8)
    decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return decoded


def normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    wrist = landmarks[0].copy()
    pts = landmarks - wrist
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def extract_features(results) -> np.ndarray:
    def hand_arr(hand_landmarks):
        if not hand_landmarks:
            return np.zeros(63, dtype=np.float32)
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        return normalize_hand(pts).flatten()

    left = hand_arr(results.left_hand_landmarks)
    right = hand_arr(results.right_hand_landmarks)
    return np.concatenate([left, right]).astype(np.float32)





class ModelRunner:
    def __init__(self, model, label_map):
        self.model = model
        self.label_map = label_map
        self.use_tflite = hasattr(model, "invoke") and hasattr(model, "get_input_details")
        self.interpreter = model if self.use_tflite else None

    def predict(self, seq: np.ndarray):
        if self.use_tflite and self.interpreter is not None:
            inp = self.interpreter.get_input_details()[0]
            out = self.interpreter.get_output_details()[0]
            self.interpreter.set_tensor(inp["index"], seq.astype(np.float32))
            self.interpreter.invoke()
            probs = self.interpreter.get_tensor(out["index"])[0]
        else:
            probs = self.model.predict(seq, verbose=0)[0]

        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {
                "label": self.label_map.get(int(i), str(i)),
                "confidence": float(probs[int(i)]),
            }
            for i in top3_idx
        ]
        best = top3[0]
        return best["label"], best["confidence"], top3


def load_model():
    global model_runner, label_map
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError("Model or label map not found in outputs/saved_models")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        raw_labels = json.load(f)

    label_map = {int(k): v for k, v in raw_labels.items()}

    if os.path.exists(TFLITE_PATH):
        try:
            from tensorflow.lite import Interpreter
            interpreter = Interpreter(model_path=TFLITE_PATH)
            interpreter.allocate_tensors()
            model_runner = ModelRunner(interpreter, label_map)
            return
        except Exception:
            pass

    import tensorflow as tf
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"AttentionLayer": AttentionLayer},
    )
    model_runner = ModelRunner(model, label_map)


def cleanup_stale_clients():
    now = time.time()
    with client_lock:
        stale = [cid for cid, state in client_buffers.items() if now - state["updated"] > CLIENT_TIMEOUT_SECONDS]
        for cid in stale:
            del client_buffers[cid]


def get_client_buffer(client_id: str):
    with client_lock:
        record = client_buffers.get(client_id)
        if record is None:
            record = {"frames": deque(maxlen=SEQ_LEN), "updated": time.time()}
            client_buffers[client_id] = record
        else:
            record["updated"] = time.time()
        return record


@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "message": "Flask backend is ready"})


@app.route("/api/predict", methods=["POST"])
def predict():
    if model_runner is None:
        return jsonify({"error": "Model not loaded"}), 503

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Missing JSON payload"}), 400

    client_id = payload.get("clientId") or "anonymous"
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "Missing image data"}), 400

    try:
        image = decode_image_data(image_data)
        if image is None:
            raise ValueError("Unable to decode image")
    except Exception as exc:
        return jsonify({"error": f"Invalid image data: {exc}"}), 400

    # Flip horizontally to match training data (model was trained on mirrored frames)
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_image)

    buffer_record = get_client_buffer(client_id)
    progress = len(buffer_record["frames"]) / SEQ_LEN

    response = {
        "status": "ok",
        "bufferProgress": round(progress, 3),
        "word": "",
        "confidence": 0.0,
        "top3": [],
        "landmarks": [],
    }

    def serialize_landmarks(results):
        landmark_list = []
        for hand_name, hand_landmarks in [("left", results.left_hand_landmarks), ("right", results.right_hand_landmarks)]:
            if not hand_landmarks:
                continue
            for idx, landmark in enumerate(hand_landmarks.landmark):
                landmark_list.append({
                    "hand": hand_name,
                    "index": idx,
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                })
        return landmark_list

    features = extract_features(results)
    response["landmarks"] = serialize_landmarks(results)

    buffer_record["frames"].append(features)
    progress = len(buffer_record["frames"]) / SEQ_LEN
    response["bufferProgress"] = round(progress, 3)

    if len(buffer_record["frames"]) < SEQ_LEN:
        return jsonify(response)

    seq = np.array(buffer_record["frames"], dtype=np.float32)[np.newaxis]
    with model_lock:
        try:
            word, confidence, top3 = model_runner.predict(seq)
        except Exception as exc:
            return jsonify({"error": f"Inference failed: {exc}"}), 500

    response["top3"] = [
        {"label": item["label"].replace("_", " "), "confidence": round(item["confidence"], 4)}
        for item in top3
    ]
    response["confidence"] = round(confidence, 4)
    if confidence >= CONF_THRESHOLD and word != "idle":
        response["word"] = word

    cleanup_stale_clients()
    return jsonify(response)


if __name__ == "__main__":
    try:
        load_model()
        print("Loaded model and label map successfully.")
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        raise
    app.run(host="0.0.0.0", port=5000, debug=False)
