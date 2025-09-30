from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import requests

app = FastAPI()

# Allow requests from anywhere (mobile app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# URLs for model and labels on GitHub
MODEL_URL = "https://github.com/jas4rica/asl-python-server/raw/main/model_unquant.tflite"
LABELS_URL = "https://github.com/jas4rica/asl-python-server/raw/main/labels.txt"

# Download model
model_path = "model_unquant.tflite"
with open(model_path, "wb") as f:
    f.write(requests.get(MODEL_URL).content)

# Download labels
labels_path = "labels.txt"
with open(labels_path, "wb") as f:
    f.write(requests.get(LABELS_URL).content)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load labels
with open(labels_path) as f:
    labels = [line.strip().split()[-1] for line in f.readlines() if line.strip()]

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = image.resize((224, 224))  # Teachable Machine default

        input_data = np.array(image, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        idx = int(np.argmax(output_data))
        confidence = float(output_data[idx])
        label = labels[idx]

        if confidence < 0.5:
            label = "None"

        return {"label": label, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}

# For testing locally (remove in production)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
