from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Retinal Disease Risk Detection API")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = tf.keras.models.load_model("rfmid_model.h5")
IMG_SIZE = 224

def is_retinal_image(img_np):
    # img_np shape: (224,224,3)
    r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]

    red_ratio = np.mean(r) / (np.mean(g) + 1e-6)
    brightness = np.mean(img_np)

    # Retinal images usually satisfy these
    if red_ratio > 1.1 and 0.1 < brightness < 0.6:
        return True
    return False

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_np = np.array(image) / 255.0
    return image_np

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_np = preprocess_image(image_bytes)

    # 🔒 Reject non-retinal images
    if not is_retinal_image(img_np):
        return {
            "prediction": "Invalid Image",
            "confidence": 0.0,
            "message": "Please upload a retinal fundus image"
        }

    img = np.expand_dims(img_np, axis=0)
    prediction = model.predict(img)[0][0]

    # 🔐 Safer thresholding
    if prediction >= 0.75:
        result = "Abnormal (High Disease Risk)"
    elif prediction <= 0.35:
        result = "Normal (Low Disease Risk)"
    else:
        result = "Uncertain – Needs Review"

    return {
        "prediction": result,
        "confidence": round(float(prediction), 4)
    }
