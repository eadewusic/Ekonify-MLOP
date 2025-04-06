from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, UploadFile, File
import logging
import os
import zipfile
from datetime import datetime
from pathlib import Path
import shutil
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_recall_fscore_support, classification_report
import numpy as np


app = FastAPI(
    title="Ekonify Retrain API",
    description="Upload a zip file containing image datasets to retrain an existing model and get the training results (accuracy, loss, etc.).",
    version="1.0"
)

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Define the path to the models directory
MODELS_BASE_PATH = Path("../models")

class ModelManager:
    def __init__(self, base_path: Path, models_base_path: Path):
        self.base_path = Path(base_path)
        self.models_base_path = Path(models_base_path)  # Path to external models directory
        self.data_dir = self.base_path / "data"
        self.uploads_dir = self.data_dir / "uploads"
        self.preprocessed_dir = self.data_dir / "preprocessed"
        
        # Create necessary directories
        for directory in [self.uploads_dir, self.preprocessed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the path for a model file."""
        return self.models_base_path / f"{model_name}.pkl"  # Using .pkl extension
    
    def validate_model_exists(self, model_name: str) -> bool:
        """Check if a model exists and list available models for debugging."""
        model_path = self.get_model_path(model_name)
        logger.info(f"Looking for model at: {model_path}")  # Debug log
        
        # Check if models directory exists
        if not self.models_base_path.exists():
            logger.error(f"Models directory not found: {self.models_base_path}")
            return False
        
        # List available models for debugging
        try:
            available_models = [f.name for f in self.models_base_path.iterdir() if f.is_file()]
            logger.info(f"Available models: {available_models}")
        except Exception as e:
            logger.error(f"Error listing models directory: {e}")
            available_models = []
        
        # Check if model exists
        exists = model_path.exists()
        logger.info(f"Model exists: {exists}")
        return exists

class DataProcessor:
    def __init__(self, manager: ModelManager):
        self.manager = manager
    
    async def save_upload(self, file: UploadFile) -> Path:
        """Save uploaded file and return its path."""  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_path = self.manager.uploads_dir / f"upload_{timestamp}.zip"
        
        try:
            content = await file.read()
            upload_path.write_bytes(content)
            return upload_path
        except Exception as e:
            logger.error(f"Error saving upload: {e}")
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    def validate_zip(self, zip_path: Path) -> bool:
        """Validate ZIP file contents.""" 
        try:
            image_count = 0
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_path in zip_ref.namelist():
                    if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):  # Ensure it's an image file
                        image_count += 1
            return image_count > 0
        except Exception as e:
            return False
    
    def process_dataset(self, zip_path: Path) -> Path:
        """Process uploaded dataset."""  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = self.manager.preprocessed_dir / f"dataset_{timestamp}"
        
        try:
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            return dataset_dir
        except Exception as e:
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            raise HTTPException(status_code=500, detail=f"Dataset processing failed: {str(e)}")

class ModelTrainer:
    def __init__(self, manager: ModelManager):
        self.manager = manager
    
    def train_model(self, model_name: str, dataset_path: Path) -> dict:
        try:
            # Load model using pickle
            model_path = self.manager.get_model_path(model_name)
            logger.info(f"Loading model from {model_path}")
            
            try:
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

            # Extract the input shape that the model expects
            input_shape = model.input_shape[1:]
            logger.info(f"Model expects input shape: {input_shape}")
            
            # Use the model's input shape to determine target size for images
            if len(input_shape) >= 3:
                target_size = (input_shape[0], input_shape[1])
            else:
                target_size = (224, 224)  # Default size
            
            logger.info(f"Using target size for images: {target_size}")
            
            # Setup data generators with the appropriate target size
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                validation_split=0.2
            )
            
            val_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )
            
            # Create generators with the correct target size
            train_generator = train_datagen.flow_from_directory(
                dataset_path,
                target_size=target_size,
                batch_size=32,
                class_mode='categorical',
                subset='training'
            )
            
            validation_generator = val_datagen.flow_from_directory(
                dataset_path,
                target_size=target_size,
                batch_size=32,
                class_mode='categorical',
                subset='validation'
            )
            
            # Recompile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train the model
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=3,
                callbacks=[ 
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=2,
                        restore_best_weights=True
                    )
                ]
            )

            # Evaluate the model on the validation data
            val_predictions = model.predict(validation_generator)
            val_true = validation_generator.classes
            val_predictions = np.argmax(val_predictions, axis=1)

            # Generate the classification report
            report = classification_report(val_true, val_predictions, output_dict=True)

            # Extract desired metrics
            precision_macro = report['macro avg']['precision']
            recall_macro = report['macro avg']['recall']
            f1_macro = report['macro avg']['f1-score']
            
            # Format the metrics to 4 decimal places
            precision_macro = round(precision_macro, 4)
            recall_macro = round(recall_macro, 4)
            f1_macro = round(f1_macro, 4)

            # Save retrained model inside 'user_retrained_models' directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            retrained_dir = self.manager.models_base_path / "user_retrained_models"
            retrained_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

            new_model_path = retrained_dir / f"{model_name}_retrained_{timestamp}.pkl"
            with open(new_model_path, 'wb') as model_file:
                pickle.dump(model, model_file)
            
            # Get the final accuracy, loss, and validation metrics, rounded to 4 decimal places
            final_accuracy = round(history.history['accuracy'][-1], 4)
            final_val_accuracy = round(history.history['val_accuracy'][-1], 4)
            final_loss = round(history.history['loss'][-1], 4)
            final_val_loss = round(history.history['val_loss'][-1], 4)
            
            return {
                "message": "Model retrained successfully",
                "model_path": str(new_model_path),
                "model_performance_metrics": {
                    "epochs": 3,
                    "final_accuracy": float(history.history['accuracy'][-1]),
                    "final_val_accuracy": float(history.history['val_accuracy'][-1]),
                    "final_loss": float(history.history['loss'][-1]),
                    "final_val_loss": float(history.history['val_loss'][-1]),
                    "precision_macro": precision_macro,
                    "recall_macro": recall_macro,
                    "f1_score_macro": f1_macro
                }
            }
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

# Initialize the managers with correct model directory
model_manager = ModelManager(Path("./"), MODELS_BASE_PATH)
data_processor = DataProcessor(model_manager)
model_trainer = ModelTrainer(model_manager)

@router.post("/upload")
async def upload_and_retrain(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Handle dataset upload and model retraining."""
    try:
        # Log model directory for debugging
        logger.info(f"Models directory path: {model_manager.models_base_path}")
        
        # Save upload
        zip_path = await data_processor.save_upload(file)
        logger.info(f"File saved to {zip_path}")
        
        # Validate ZIP contents
        if not data_processor.validate_zip(zip_path):
            raise HTTPException(status_code=400, detail="Invalid dataset in ZIP file")
        
        # Process dataset
        dataset_path = data_processor.process_dataset(zip_path)
        logger.info(f"Dataset processed at {dataset_path}")
        
        # Check if model exists
        if not model_manager.validate_model_exists("baseline_cnn"):
            # More detailed error message
            try:
                available_models = [f.name for f in model_manager.models_base_path.iterdir() if f.is_file()]
                models_list = ", ".join(available_models) if available_models else "none found"
            except Exception:
                models_list = "unable to list models"
                
            raise HTTPException(
                status_code=404, 
                detail=f"Model 'baseline_cnn' not found. Models directory: {model_manager.models_base_path}, Available models: {models_list}"
            )

        # Train model
        results = model_trainer.train_model("baseline_cnn", dataset_path)
        
        # Clean up the uploaded zip file
        os.remove(zip_path)
        
        return {"message": "Model retrained successfully", "results": results}
    
    except HTTPException as e:
        logger.error(f"HTTPException: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Ekonify Retrain API is running!"}

# Add the router to the FastAPI app
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("retrain:app", host="127.0.0.1", port=8000, reload=True)
