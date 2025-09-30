from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = FastAPI()

# Allow requests from anywhere (mobile app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Load labels
with open("labels.txt") as f:
    labels = [line.strip().split()[-1] for line in f.readlines() if line.strip()]

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = image.resize((224, 224))  # Teachable Machine default

        input_data = np.array(image, dtype=np.float32)
        input_data = input_data / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        idx = np.argmax(output_data)
        confidence = float(output_data[idx])
        label = labels[idx]

        if confidence < 0.5:
            label = "None"

        return {"label": label, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}


# For local testing only
# uvicorn.run(app, host="0.0.0.0", port=8000)
