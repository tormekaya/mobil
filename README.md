# Mobil Görüntü Sınıflandırma API'si

Bu proje, mobil uygulamalardan gelen görüntüleri sınıflandırmak üzere eğitilmiş `Keras (.keras)` ve `H5 (.h5)` modellerini sunucuda çalıştırır. Flask tabanlı bu sunucu, Google Cloud Run üzerinde host edilir. Mobil istemciler sunucuya HTTP POST istekleri göndererek tahmin sonuçlarını alabilir.

## 🚀 Özellikler

- Flask REST API
- Keras ve TensorFlow modelleriyle sınıflandırma
- Google Cloud Storage entegrasyonu
- Mobil uygulamalarla uyumlu HTTP arayüz
- Docker desteği
- CORS etkin

## 📁 Dosya Yapısı

- `app.py`: API ana uygulama dosyası
- `requirements.txt`: Python bağımlılıkları
- `Dockerfile`: Docker imajı oluşturmak için yapılandırma
- `cloud.txt`: Google Cloud Run deploy komutu

## 🐳 Docker ile Çalıştırma

Proje, aşağıdaki `Dockerfile` içeriğiyle paketlenmiştir:

```Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD exec gunicorn --bind :8080 --workers 1 --threads 8 app:app
