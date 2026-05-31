import base64
import json
import os
import sys
import threading
import time
from collections import deque

import cv2
mp_solutions = None
mp_import_error = None
try:
    from mediapipe.python import solutions as mp_solutions
except Exception as exc:
    mp_import_error = exc
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
genai = None
genai_import_error = None
try:
    import google.generativeai as genai
except Exception as exc:
    genai_import_error = exc

load_dotenv()

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

mp_holistic = mp_solutions.holistic if mp_solutions is not None else None
holistic = None
holistic_init_error = mp_import_error


def init_holistic():
    global holistic, holistic_init_error
    if holistic is not None:
        return holistic
    if holistic_init_error is not None:
        return None
    if mp_holistic is None:
        holistic_init_error = RuntimeError("mediapipe is unavailable")
        return None

    try:
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,  # Lightweight model for faster two-hand detection
        )
        return holistic
    except Exception as exc:
        holistic_init_error = exc
        return None


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

    import importlib
    import tensorflow as tf

    keras_functional = importlib.import_module("keras.src.models.functional").Functional
    custom_objects = {
        "Functional": keras_functional,
        "AttentionLayer": AttentionLayer,
        "LSTM": tf.keras.layers.LSTM,
        "Bidirectional": tf.keras.layers.Bidirectional,
        "Dropout": tf.keras.layers.Dropout,
        "Dense": tf.keras.layers.Dense,
        "BatchNormalization": tf.keras.layers.BatchNormalization,
        "InputLayer": tf.keras.layers.InputLayer,
    }

    def _strip_time_major(config_obj):
        if isinstance(config_obj, dict):
            config_obj.pop("time_major", None)
            for value in config_obj.values():
                _strip_time_major(value)
        elif isinstance(config_obj, list):
            for item in config_obj:
                _strip_time_major(item)

    def _load_legacy_h5_model():
        import h5py

        with h5py.File(MODEL_PATH, "r") as h5f:
            raw_config = h5f.attrs["model_config"]
            config = json.loads(raw_config if isinstance(raw_config, str) else raw_config.decode("utf-8"))

        _strip_time_major(config)
        return tf.keras.models.model_from_json(json.dumps(config), custom_objects=custom_objects)

    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    except Exception:
        model = _load_legacy_h5_model()
        model.load_weights(MODEL_PATH)

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


def _simple_fallback_sentence(words):
    if not words:
        return ""
    lw = [w.lower() for w in words]
    # Heuristic rules for common intents
    if "doctor" in lw and "help" in lw:
        return "I need a doctor, please help me."
    if "help" in lw and "please" in lw:
        return "Please help me."
    if "help" in lw:
        return "Help me, please."
    if "thank" in " ".join(lw) or "thank_you" in lw:
        return "Thank you."
    # Generic fallback: join and punctuate
    s = " ".join(words).strip()
    if not s:
        return ""
    if not s.endswith('.') and not s.endswith('!') and not s.endswith('?'):
        s = s + '.'
    return s[0].upper() + s[1:]


@app.route("/api/generate-sentence", methods=["POST"])
def generate_sentence():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Missing JSON payload"}), 400

    words = payload.get("words")
    if not isinstance(words, list) or not words:
        return jsonify({"error": "`words` must be a non-empty list"}), 400

    # Try to call Gemini using API key from .env
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        if genai is None:
            raise RuntimeError(f"Gemini client unavailable: {genai_import_error}")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
        You are an assistive communication system for non-verbal sign language users.

        Your task is to infer the user's most likely intended message from detected sign language words.

        Detected words:
        {' '.join(words)}

        Rules:
        - Infer the user's intended meaning.
        - Convert the words into ONE natural sentence.
        - Prioritize practical communication needs.
        - Keep the sentence concise and clear.
        - Do not explain your reasoning.
        - Do not add unnecessary details.
        - Return ONLY the sentence.

        Examples:

        help doctor please
        → I need a doctor, please help me.

        water please
        → Could I please have some water?

        bathroom
        → I need to use the bathroom.

        hungry food
        → I am hungry and would like some food.

        thank you
        → Thank you.

        Now generate the sentence.
        """

        response = model.generate_content(prompt)
        sentence = getattr(response, "text", None)
        if sentence:
            sentence = sentence.strip()
        if not sentence:
            raise ValueError("Empty response from Gemini")

        return jsonify({"sentence": sentence})

    except Exception as exc:
        # Fallback behavior: return a deterministic heuristic-based sentence
        fallback = _simple_fallback_sentence(words)
        return jsonify({"sentence": fallback, "fallback": True, "error": str(exc)}), 200


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

    hol = init_holistic()
    if hol is None:
        msg = "Mediapipe Holistic initialization failed"
        if holistic_init_error is not None:
            msg = f"Mediapipe Holistic initialization failed: {holistic_init_error}"
        return jsonify({"error": msg}), 503

    # Flip horizontally to match training data (model was trained on mirrored frames)
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hol.process(rgb_image)

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
