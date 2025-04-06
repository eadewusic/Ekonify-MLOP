from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import numpy as np
import io
import pickle

app = FastAPI(
    title="Ekonify Predict API",
    description="Upload an image to get the predicted waste class and confidence score.",
    version="1.0"
)

# Load trained model
MODEL_PATH = "../models/baseline_cnn.pkl"

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)  # Use joblib if it's an sklearn model
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Define class names (update this list based on trained model's classes)
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 
               'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# Prediction function
def predict(image: Image.Image):
    try:
        image = image.resize((224, 224)).convert("RGB")  # Resize & ensure 3 channels
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(predictions[0]))

        return predicted_class, confidence
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/predict/", summary="Predict Waste Type from Image", description="Upload an image file and receive the predicted waste type along with the confidence score.")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {e}")

    predicted_class, confidence = predict(image)
    return {
        "prediction": predicted_class,
        "confidence": f"{round(confidence * 100, 2)}%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("predict:app", host="127.0.0.1", port=8000, reload=True)
