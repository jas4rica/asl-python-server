from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import requests

# ===== FastAPI App =====
app = FastAPI()

# Allow requests from Flutter or any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Model and Labels URLs (already uploaded to GitHub) =====
MODEL_URL = "https://github.com/jas4rica/asl-python-server/raw/main/keras_model.h5"
LABELS_URL = "https://github.com/jas4rica/asl-python-server/raw/main/labels.txt"

# Load model from URL
model_bytes = requests.get(MODEL_URL).content
model_file = io.BytesIO(model_bytes)
model = tf.keras.models.load_model(model_file)

# Load labels from URL
labels_bytes = requests.get(LABELS_URL).content
labels_file = io.BytesIO(labels_bytes)
labels = [line.strip().decode("utf-8").split()[-1] for line in labels_file.readlines() if line.strip()]

# ===== Prediction Endpoint =====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = image.resize((224, 224))  # Teachable Machine default

        # Normalize and expand dims
        input_data = np.array(image, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        # Run model
        predictions = model.predict(input_data)
        idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][idx])
        label = labels[idx] if confidence >= 0.5 else "None"

        return {"label": label, "confidence": round(confidence * 100, 2)}
    except Exception as e:
        return {"error": str(e)}

# ===== Optional Test Web UI =====
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <body>
            <h1>ASL Recognition Server</h1>
            <p>Send an image via POST /predict for recognition.</p>
        </body>
    </html>
    """
