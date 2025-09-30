from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import requests
import io
import asyncio

app = FastAPI()

# Allow cross-origin para sa Flutter/Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Load Keras model at labels mula sa GitHub =====
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

# ===== WebSocket para sa live detection =====
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    last_prediction = {"label": "None", "confidence": 0.0}
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_frame = cv2.resize(rgb_frame, (224, 224))

            if frame_count % 3 == 0:  # predict every 3 frames para di masyado mabagal
                input_data = np.expand_dims(small_frame.astype(np.float32)/255.0, axis=0)
                predictions = model.predict(input_data)
                idx = int(np.argmax(predictions[0]))
                confidence = float(predictions[0][idx])
                label = labels[idx] if confidence >= 0.5 else "None"
                last_prediction = {"label": label, "confidence": round(confidence*100,2)}

            # Draw MediaPipe landmarks
            results = hands_detector.process(rgb_frame)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Send JSON result via WebSocket
            await websocket.send_text(f'{last_prediction}')

            frame_count += 1
            await asyncio.sleep(0.05)  # ~20 FPS

    except Exception as e:
        await websocket.send_text(f'{{"error":"{str(e)}"}}')
    finally:
        cap.release()
        await websocket.close()


# ===== Simple Web UI para sa live detection =====
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>Si-Lang Recognize</title>
        <style>
            body { text-align: center; font-family: Arial, sans-serif; background-color: #f5f5f5; }
            h1 { margin-top: 30px; color: #222; }
            #detected-box { 
                display: inline-block; 
                margin-top: 20px; 
                padding: 20px 40px; 
                background-color: black; 
                color: white; 
                font-size: 2em; 
                border-radius: 10px;
            }
        </style>
    </head>
    <body>
        <h1>Si-Lang Recognize</h1>
        <div id="detected-box">
            Detected Sign: <span id="sign">None</span> | Confidence: <span id="conf">0%</span>
        </div>

        <script>
            let ws = new WebSocket("ws://YOUR_SERVER_IP:8000/ws");
            ws.onmessage = function(event) {
                try {
                    let data = JSON.parse(event.data);
                    document.getElementById("sign").innerText = data.label;
                    document.getElementById("conf").innerText = data.confidence + "%";
                } catch (e) {
                    console.error("Error parsing WS message:", e);
                }
            };
            ws.onopen = function() { console.log("WebSocket connected!"); };
            ws.onclose = function() { console.log("WebSocket disconnected!"); };
        </script>
    </body>
    </html>
    """
