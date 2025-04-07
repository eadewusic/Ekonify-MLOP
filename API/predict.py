from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import numpy as np
import io
import pickle
from database import Database
import logging
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Ekonify Predict API",
    description="API for image prediction with automatic result storage",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained model
MODEL_PATH = "../models/baseline_cnn.pkl"
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

# Class names
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# Database connection
db = Database()

def predict(image: Image.Image):
    try:
        image = image.resize((224, 224)).convert("RGB")
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(predictions[0]))

        return predicted_class, confidence
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    feedback: str = None
):
    """Predict image class and automatically store in database"""
    try:
        # Process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Get prediction
        predicted_class, confidence = predict(image)
        
        # Always store in database
        try:
            image_id = db.insert_image(
                file_data=image_bytes,
                filename=file.filename,
                label=predicted_class,
                metadata={
                    "confidence": confidence,
                    "feedback": feedback or "",
                    "prediction": predicted_class
                }
            )
            logger.info(f"Stored prediction in database with ID: {image_id}")
            stored_in_db = True
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
            stored_in_db = False
        
        return {
            "prediction": predicted_class,
            "confidence": f"{round(confidence * 100, 2)}%",
            "stored_in_db": stored_in_db
        }
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image format.")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("predict:app", host="127.0.0.1", port=8000, reload=True)
