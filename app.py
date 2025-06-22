from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from google.cloud import storage
import os
import tempfile

app = Flask(__name__)

# GCS ayarı
BUCKET_NAME = 'mobilprojesi'
BODY_MODEL_FILE = 'hybrid_body.keras'
WING_MODEL_FILE = 'mobilenetv2_wing.h5'

# GCS'ten modeli geçici olarak indir ve yükle
def download_model(model_name):
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(model_name)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    blob.download_to_filename(temp_file.name)
    model = tf.keras.models.load_model(temp_file.name)
    return model

body_model = download_model(BODY_MODEL_FILE)
wing_model = download_model(WING_MODEL_FILE)

labels = ['AE', 'AL', 'JA', 'KO']  # Örnek sınıf etiketleri

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Resim bulunamadı'}), 400

    image_file = request.files['image']
    region = request.form.get('region')  # 'body' veya 'wing'

    if region not in ['body', 'wing']:
        return jsonify({'error': 'Geçersiz region'}), 400

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
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
