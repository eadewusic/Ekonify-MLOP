from fastapi import FastAPI, UploadFile, File, HTTPException
from database import Database
from bson import ObjectId
from typing import List
import os

app = FastAPI()
db = Database()

@app.get("/")
def root():
    return {"message": "Welcome to Ekonify"}

@app.post("/predict")
async def upload_image(file: UploadFile = File(...), label: str = None):
    file_content = await file.read()
    metadata = {"filename": file.filename, "content_type": file.content_type}
    image_id = db.insert_image(file_content, label, metadata)
    return {"image_id": image_id, "message": "Image uploaded successfully"}

@app.get("/images/{image_id}")
def get_image(image_id: str):
    image = db.retrieve_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return image

@app.put("/images/{image_id}")
def update_image(image_id: str, label: str = None):
    updated_data = {"label": label}
    if db.update_image(image_id, updated_data):
        return {"message": "Image updated successfully"}
    raise HTTPException(status_code=400, detail="Failed to update image")

@app.delete("/images/{image_id}")
def delete_image(image_id: str):
    if db.delete_image(image_id):
        return {"message": "Image deleted successfully"}
    raise HTTPException(status_code=400, detail="Failed to delete image")

@app.post("/retraining/")
async def retraining(file: UploadFile = File(...), label: str = None, feedback: str = None):
    file_content = await file.read()
    retrain_id = db.insert_for_retraining(file_content, label, feedback)
    return {"retrain_id": retrain_id, "message": "Image added for retraining"}

@app.get("/retraining/")
def get_retraining_images():
    images = db.retrieve_for_retraining()
    return {"images": images}

@app.delete("/retraining/{image_id}")
def delete_retraining_image(image_id: str):
    if db.delete_retraining_image(image_id):
        return {"message": "Retraining image deleted successfully"}
    raise HTTPException(status_code=400, detail="Failed to delete retraining image")
