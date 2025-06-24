# 📱 Mobile Image Classification API

This project runs pre-trained `Keras (.keras)` and `H5 (.h5)` models on a server for image classification requests coming from mobile applications. The Flask-based API is hosted on **Google Cloud Run**, and mobile clients can send HTTP POST requests to receive predictions.

## 🚀 Features

- Flask REST API
- Image classification with Keras and TensorFlow models
- Google Cloud Storage integration
- Mobile-friendly HTTP interface
- Docker support
- CORS enabled

## 📁 Project Structure

- `app.py`: Main API application file
- `requirements.txt`: Python dependencies
- `Dockerfile`: Docker image configuration
- `cloud.txt`: Google Cloud Run deploy command

## 🐳 Running with Docker

The project is containerized with the following Dockerfile:

```Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD exec gunicorn --bind :8080 --workers 1 --threads 8 app:app
```
