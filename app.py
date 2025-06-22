from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Modelleri yükle
body_model = tf.keras.models.load_model('hybrid_body.keras')
wing_model = tf.keras.models.load_model('mobilenetv2_wing.h5')

labels = ['AE', 'AL', 'JA', 'KO']

# Görüntü ön işleme
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((384, 384))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return 'Sivrisinek Tahmin API'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Resim dosyası \"image\" parametresi ile yüklenmeli.'}), 400

    region = request.form.get('region')
    if region not in ['body', 'wing']:
        return jsonify({'error': 'region parametresi \"body\" veya \"wing\" olmalı.'}), 400

    try:
        image_bytes = request.files['image'].read()
        img_array = preprocess_image(image_bytes)

        # Uygun modeli seç
        model = body_model if region == 'body' else wing_model

        prediction = model.predict(img_array)[0]
        result = [
            {"label": labels[i], "confidence": float(prediction[i])}
            for i in range(len(labels))
        ]
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
