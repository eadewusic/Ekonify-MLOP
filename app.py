from fastapi import FastAPI, File, UploadFile
from PIL import Image
import tensorflow as tf
import numpy as np
import io

app = FastAPI()

# Load your trained Keras model
MODEL_PATH = "./models/baseline_cnn.keras"
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()  # Optional, to confirm the model loads correctly

# Define your class names (update this list based on your trained model's classes)
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']  # Replace with your actual class names

# Prediction function
def predict(image: Image.Image):
    # Preprocess the image
    image = image.resize((224, 224))  # Match your model's input size
    image = image.convert("RGB")
    
    # Convert the image to a numpy array
    image_array = np.array(image) / 255.0  # Normalize the image to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]  # Get class name from index
    
    return predicted_class


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    prediction = predict(image)
    return {"prediction": prediction}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
