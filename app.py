from flask import Flask, request, jsonify
from google.cloud import storage
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# GCS bucket adı ve dosya yolları
BUCKET_NAME = "mobilprojesi"
BODY_MODEL_BLOB = "hybrid_body.keras"
WING_MODEL_BLOB = "mobilenetv2_wing.h5"

BODY_MODEL_PATH = "/tmp/hybrid_body.keras"
WING_MODEL_PATH = "/tmp/mobilenetv2_wing.h5"

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """GCS'den dosyayı indir"""
    if os.path.exists(destination_file_name):
        print(f"{destination_file_name} zaten mevcut, tekrar indirme.")
        return
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"{destination_file_name} indirildi.")

# Model dosyalarını indir ve yükle
download_blob(BUCKET_NAME, BODY_MODEL_BLOB, BODY_MODEL_PATH)
download_blob(BUCKET_NAME, WING_MODEL_BLOB, WING_MODEL_PATH)

body_model = tf.keras.models.load_model(BODY_MODEL_PATH)
wing_model = tf.keras.models.load_model(WING_MODEL_PATH)

labels = ['AE', 'AL', 'JA', 'KO']

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Resim bulunamadı'}), 400

    image_file = request.files['image']
    region = request.form.get('region')

    if region not in ['body', 'wing']:
        return jsonify({'error': 'Region değeri eksik veya hatalı'}), 400

    try:
        image = Image.open(image_file.stream).convert("RGB")
        image = image.resize((384, 384))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        model = body_model if region == 'body' else wing_model

        preds = model.predict(image_array)[0]
        results = [{"label": labels[i], "confidence": float(preds[i])} for i in range(len(labels))]

        return jsonify({"prediction": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "🧪 Model API çalışıyor."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
