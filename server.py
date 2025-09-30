from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import io
import asyncio
import base64

app = FastAPI()

# Allow cross-origin for Flutter/web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Load Keras model and labels from GitHub =====
MODEL_URL = "https://github.com/jas4rica/asl-python-server/raw/main/keras_model.h5"
LABELS_URL = "https://github.com/jas4rica/asl-python-server/raw/main/labels.txt"

model_bytes = requests.get(MODEL_URL).content
model_file = io.BytesIO(model_bytes)
model = tf.keras.models.load_model(model_file)

labels_bytes = requests.get(LABELS_URL).content
labels_file = io.BytesIO(labels_bytes)
labels = [line.strip().decode("utf-8").split()[-1] for line in labels_file.readlines() if line.strip()]

# ===== MediaPipe Hands =====
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ===== WebSocket for live detection =====
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Use OpenCV to access webcam (0) or a virtual camera if deployed
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe hand detection
            results = hands_detector.process(rgb_frame)

            # Draw landmarks if any
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Preprocess frame for Keras model
            resized = cv2.resize(rgb_frame, (224, 224))
            input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

            # Predict
            predictions = model.predict(input_data)
            idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][idx])
            label = labels[idx] if confidence >= 0.5 else "None"

            # Send JSON result via WebSocket
            await websocket.send_text(f'{{"label":"{label}","confidence":{round(confidence*100,2)}}}')

            await asyncio.sleep(0.05)  # ~20 FPS

    except Exception as e:
        await websocket.send_text(f'{{"error":"{str(e)}"}}')
    finally:
        cap.release()
        await websocket.close()


# ===== Simple Web UI for testing =====
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
    <title>ASL Live Recognition</title>
    </head>
    <body>
        <h1>ASL Live Recognition (MediaPipe + Keras)</h1>
        <p>Connect via WebSocket to <code>/ws</code> for live detection.</p>
        <p>Use your Flutter app or a browser WebSocket client to receive <strong>label + confidence</strong> updates.</p>
    </body>
    </html>
    """
