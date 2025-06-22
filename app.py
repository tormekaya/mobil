from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

from google.cloud import storage

app = Flask(__name__)

# Bucket ve model adları
BUCKET_NAME = 'mobilprojesi'
BODY_MODEL_FILE = 'hybrid_body.keras'
WING_MODEL_FILE = 'mobilenetv2_wing.h5'
LOCAL_BODY_PATH = f'/tmp/{BODY_MODEL_FILE}'
LOCAL_WING_PATH = f'/tmp/{WING_MODEL_FILE}'

# Google Cloud Storage istemcisi
storage_client = storage.Client()

# Modeli Cloud Storage’tan indir (eğer yerelde yoksa)
def download_model_from_gcs(file_name, local_path):
    if not os.path.exists(local_path):
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(file_name)
        blob.download_to_filename(local_path)
        print(f"{file_name} indirildi.")

# Modelleri indir ve yükle
download_model_from_gcs(BODY_MODEL_FILE, LOCAL_BODY_PATH)
download_model_from_gcs(WING_MODEL_FILE, LOCAL_WING_PATH)

body_model = tf.keras.models.load_model(LOCAL_BODY_PATH)
wing_model = tf.keras.models.load_model(LOCAL_WING_PATH)

labels = ['AE', 'AL', 'JA', 'KO']  # Sınıf etiketleri

@app.route('/predict', methods=['POST'])
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
