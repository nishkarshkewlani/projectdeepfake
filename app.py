from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__, static_folder=".")
CORS(app)

MODEL_PATH = "/workspaces/deepfake/resnet_deepfake_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model missing at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # MATCHES YOUR MODEL INPUT SHAPE
    img = img.resize((128, 128))

    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_bytes = file.read()

        if not file_bytes:
            return jsonify({"error": "Empty file"}), 400

        img = preprocess_image(file_bytes)

        pred = model.predict(img, verbose=0)

        score = float(pred[0][0]) if len(pred.shape) > 1 else float(pred[0])

        label = "FAKE" if score >= 0.5 else "REAL"
        confidence = score if score >= 0.5 else 1 - score

        return jsonify({
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
