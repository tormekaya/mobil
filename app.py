from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import storage
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# GCS yapılandırması
BUCKET_NAME = "mobilprojesi"
MODELS = {
    "body": {
        "blob": "hybrid_body.keras",
        "path": "/tmp/hybrid_body.keras"
    },
    "wing": {
        "blob": "mobilenetv2_wing.h5",
        "path": "/tmp/mobilenetv2_wing.h5"
    }
}

labels = ['AE', 'AL', 'JA', 'KO']
models = {}

def download_blob_if_needed(bucket_name, source_blob_name, destination_file_name):
    """GCS'den dosyayı indir (eğer yoksa)."""
    if os.path.exists(destination_file_name):
        print(f"{destination_file_name} zaten mevcut.")
        return
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"{destination_file_name} indirildi.")
    except Exception as e:
        print(f"GCS'den model indirme hatası: {e}")
        raise

def load_models():
    """Model dosyalarını indir ve yükle."""
    for key, info in MODELS.items():
        download_blob_if_needed(BUCKET_NAME, info["blob"], info["path"])
        models[key] = tf.keras.models.load_model(info["path"])
        print(f"{key} modeli yüklendi.")

# Modelleri yükle
load_models()

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Resim bulunamadı'}), 400

    region = request.form.get('region')
    if region not in models:
        return jsonify({'error': 'Region (body/wing) değeri eksik veya geçersiz'}), 400

    try:
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert("RGB")
        image = image.resize((384, 384))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        preds = models[region].predict(image_array)[0]
        results = [
            {"label": labels[i], "confidence": float(preds[i])}
            for i in range(len(labels))
        ]
        return jsonify({"prediction": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "✅ Model API çalışıyor."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
