from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
import logging
import os
import zipfile
from datetime import datetime
from pathlib import Path
import shutil
import pickle
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
from database import Database
import io
from PIL import Image

app = FastAPI(
    title="Ekonify Retrain API",
    description="Retrain models using uploaded datasets or existing database data.",
    version="2.0"
)

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Define paths
MODELS_BASE_PATH = Path("../models")
TEMP_DATA_PATH = Path("./temp_data")
TEMP_DATA_PATH.mkdir(exist_ok=True)

class ModelManager:
    def __init__(self, base_path: Path, models_base_path: Path):
        self.base_path = Path(base_path)
        self.models_base_path = Path(models_base_path)  # Path to external models directory
        self.data_dir = self.base_path / "data"
        self.uploads_dir = self.data_dir / "uploads"
        self.preprocessed_dir = self.data_dir / "preprocessed"
        self.db = Database()
        
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
    
    def get_latest_retrained_model(self, base_model_name: str):
        """Get the latest retrained model from database."""
        try:
            result = self.db.get_latest_retrained_model(base_model_name)
            if not result:
                logger.warning(f"No retrained model found for {base_model_name}")
                return None
            
            # The database now returns both the model and the model_data
            model, model_data = result
            logger.info(f"Loaded retrained model from database: {model_data['model_name']}")
            return model, model_data
        except Exception as e:
            logger.error(f"Failed to load model from database: {e}")
            return None

class DataProcessor:
    def __init__(self, manager: ModelManager):
        self.manager = manager
        self.db = Database()
    
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
    
    async def save_dataset_to_db(self, zip_path: Path) -> str:
        """Save the entire dataset to database for future use."""
        try:
            # Read zip file content
            with open(zip_path, 'rb') as f:
                zip_content = f.read()
                
            # Save to database - store_dataset now uses GridFS
            dataset_id = self.db.store_dataset(zip_content)
            logger.info(f"Dataset saved to database with ID: {dataset_id}")
            return dataset_id
        except Exception as e:
            logger.error(f"Error saving dataset to database: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save dataset to database: {str(e)}")

    def prepare_dataset_from_db(self):
        """Prepare dataset from all database-stored images for training"""
        try:
            # Get previously stored datasets
            datasets = self.db.get_all_datasets(limit=10)  # Get latest datasets
            if not datasets:
                raise HTTPException(status_code=404, detail="No datasets available for retraining")
            
            # Create organized directory structure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_path = TEMP_DATA_PATH / f"dataset_{timestamp}"
            dataset_path.mkdir()
            
            used_dataset_ids = []
            
            # Extract and process each dataset
            for dataset in datasets:
                try:
                    # Create a temporary file to store the zip
                    temp_zip = TEMP_DATA_PATH / f"temp_{datetime.now().timestamp()}.zip"
                    with open(temp_zip, 'wb') as f:
                        f.write(dataset["zip_data"])
                    
                    # Extract this dataset to our target directory
                    with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                        zip_ref.extractall(dataset_path)
                    
                    # Clean up temp zip
                    os.remove(temp_zip)
                    
                    # Track this dataset as used
                    if "_id" in dataset:
                        used_dataset_ids.append(dataset["_id"])
                except Exception as e:
                    logger.error(f"Error processing dataset {dataset.get('_id', 'unknown')}: {e}")
                    continue
            
            # Make sure we extracted at least some data
            if not os.listdir(dataset_path):
                raise HTTPException(status_code=500, detail="Failed to extract any usable data from datasets")
                
            logger.info(f"Prepared combined dataset from {len(used_dataset_ids)} stored datasets")
            return dataset_path, used_dataset_ids
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise HTTPException(status_code=500, detail=f"Dataset preparation failed: {str(e)}")

class ModelTrainer:
    def __init__(self, manager: ModelManager):
        self.manager = manager
        self.db = Database()
    
    def train_model(self, model_name: str, dataset_path: Path, epochs: int = 3) -> dict:
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
                epochs=epochs,
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

            # Create timestamp for model naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_model_name = f"{model_name}_retrained_{timestamp}"

            # Save retrained model to disk (for legacy support)
            retrained_dir = self.manager.models_base_path / "user_retrained_models"
            retrained_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

            new_model_path = retrained_dir / f"{new_model_name}.pkl"
            with open(new_model_path, 'wb') as model_file:
                pickle.dump(model, model_file)
            
            # Collect metrics for return
            metrics_data = {
                "epochs": epochs,
                "final_accuracy": float(history.history['accuracy'][-1]),
                "final_val_accuracy": float(history.history['val_accuracy'][-1]),
                "final_loss": float(history.history['loss'][-1]),
                "final_val_loss": float(history.history['val_loss'][-1]),
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_score_macro": f1_macro
            }
            
            # Serialize model for database storage
            model_binary = pickle.dumps(model)
            
            # Store model in database - now uses GridFS
            model_id = self.db.store_model(
                model_binary, 
                new_model_name, 
                base_model_name=model_name,
                metrics=metrics_data
            )
            
            return {
                "message": "Model retrained successfully",
                "model_path": str(new_model_path),
                "model_id": model_id,
                "model_name": new_model_name,
                "model_performance_metrics": metrics_data
            }
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")
    
    def continue_training_from_db(self, base_model_name: str, additional_epochs: int = 1) -> dict:
        """Continue training the latest retrained model from database."""
        try:
            # Get the latest retrained model from database
            result = self.manager.get_latest_retrained_model(base_model_name)
            if not result:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No retrained model found for {base_model_name}. Please retrain with dataset first."
                )
                
            model, model_data = result
            
            # Convert ObjectId to string for proper serialization later
            if "_id" in model_data and hasattr(model_data["_id"], "__str__"):
                model_data["_id"] = str(model_data["_id"])
            
            # Get the stored datasets
            dataset_path, used_dataset_ids = data_processor.prepare_dataset_from_db()
            
            # Extract metrics from previously trained model
            previous_metrics = model_data.get("metrics", {})
            previous_epochs = previous_metrics.get("epochs", 0)
            
            # Set up data generators same as in train_model function
            input_shape = model.input_shape[1:]
            if len(input_shape) >= 3:
                target_size = (input_shape[0], input_shape[1])
            else:
                target_size = (224, 224)  # Default size
            
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
            
            # Continue training for additional epochs
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=additional_epochs,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=1,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Evaluate the model
            val_predictions = model.predict(validation_generator)
            val_true = validation_generator.classes
            val_predictions = np.argmax(val_predictions, axis=1)
            
            # Generate classification report
            report = classification_report(val_true, val_predictions, output_dict=True)
            
            # Extract metrics
            precision_macro = round(report['macro avg']['precision'], 4)
            recall_macro = round(report['macro avg']['recall'], 4)
            f1_macro = round(report['macro avg']['f1-score'], 4)
            
            # Create timestamp for model naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_model_name = f"{base_model_name}_retrained_extended_{timestamp}"
            
            # Save the model to disk
            retrained_dir = self.manager.models_base_path / "user_retrained_models"
            retrained_dir.mkdir(parents=True, exist_ok=True)
            
            new_model_path = retrained_dir / f"{new_model_name}.pkl"
            with open(new_model_path, 'wb') as model_file:
                pickle.dump(model, model_file)
            
            # Collect metrics
            metrics_data = {
                "epochs": previous_epochs + additional_epochs,  # Total epochs including previous training
                "additional_epochs": additional_epochs,
                "final_accuracy": float(history.history['accuracy'][-1]),
                "final_val_accuracy": float(history.history['val_accuracy'][-1]),
                "final_loss": float(history.history['loss'][-1]),
                "final_val_loss": float(history.history['val_loss'][-1]),
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_score_macro": f1_macro,
                "previous_model_id": str(model_data.get("_id", ""))  # Convert ObjectId to string
            }
            
            # Serialize model for database storage
            model_binary = pickle.dumps(model)
            
            # Store model in database - now uses GridFS
            model_id = self.db.store_model(
                model_binary, 
                new_model_name, 
                base_model_name=base_model_name,
                metrics=metrics_data
            )
            
            # Clean up temporary dataset files
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
            
            return {
                "message": f"Model training extended by {additional_epochs} epochs successfully",
                "model_path": str(new_model_path),
                "model_id": model_id,
                "model_name": new_model_name,
                "total_epochs": previous_epochs + additional_epochs,
                "model_performance_metrics": metrics_data
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Continue training error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Continue training failed: {str(e)}")

# Initialize components
model_manager = ModelManager(Path("./"), MODELS_BASE_PATH)
data_processor = DataProcessor(model_manager)
model_trainer = ModelTrainer(model_manager)

# API Endpoints
@router.post("/upload")
async def upload_and_retrain(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    model_name: str = "baseline_cnn"
):
    """Upload a zip with dataset, retrain model, save both to database"""
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
        if not model_manager.validate_model_exists(model_name):
            # More detailed error message
            try:
                available_models = [f.name for f in model_manager.models_base_path.iterdir() if f.is_file()]
                models_list = ", ".join(available_models) if available_models else "none found"
            except Exception:
                models_list = "unable to list models"
                
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not found. Models directory: {model_manager.models_base_path}, Available models: {models_list}"
            )

        # Train model
        results = model_trainer.train_model(model_name, dataset_path)
        
        # Save the dataset to database for future retraining
        dataset_id = await data_processor.save_dataset_to_db(zip_path)
        
        # Clean up local files
        if os.path.exists(zip_path):
            os.remove(zip_path)
        
        if background_tasks:
            background_tasks.add_task(shutil.rmtree, dataset_path)
        else:
            shutil.rmtree(dataset_path)
        
        return {
            "message": "Model retrained successfully and dataset saved to database",
            "results": results,
            "dataset_id": dataset_id
        }
    
    except HTTPException as e:
        logger.error(f"HTTPException: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/continue-training")
async def continue_training(
    model_name: str = "baseline_cnn",
    additional_epochs: int = 1,
    background_tasks: BackgroundTasks = None
):
    """Retrieve latest retrained model from DB and continue training with additional epochs"""
    try:
        # Continue training with the latest retrained model
        results = model_trainer.continue_training_from_db(model_name, additional_epochs)
        
        # Convert any ObjectId to string before returning
        if isinstance(results, dict):
            # Helper function to convert ObjectId to string recursively
            def convert_objectid(item):
                if isinstance(item, dict):
                    return {k: convert_objectid(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [convert_objectid(i) for i in item]
                elif str(type(item)) == "<class 'bson.objectid.ObjectId'>":
                    return str(item)
                else:
                    return item
            
            # Convert all ObjectId instances to strings
            results = convert_objectid(results)
        
        return results
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Continue training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Continue training failed: {str(e)}")

@router.post("/retrain-from-db")
async def retrain_from_database(
    model_name: str = "baseline_cnn",
    background_tasks: BackgroundTasks = None
):
    """Retrain model using all images previously stored in database datasets"""
    try:
        # Prepare dataset from DB using previously collected data
        dataset_path, used_dataset_ids = data_processor.prepare_dataset_from_db()
        
        # Check if model exists
        if not model_manager.validate_model_exists(model_name):
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
        
        # Train model from scratch using all stored datasets
        results = model_trainer.train_model(model_name, dataset_path)
        
        # Record training session
        training_id = data_processor.db.record_training_session(
            model_name, 
            results["model_performance_metrics"],
            used_dataset_ids
        )
        
        # Clean up temporary dataset files
        if background_tasks:
            background_tasks.add_task(shutil.rmtree, dataset_path)
        else:
            shutil.rmtree(dataset_path)
        
        return {
            "message": "Model retrained successfully using all stored datasets",
            "results": results,
            "training_session_id": training_id,
            "datasets_used": len(used_dataset_ids)
        }
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Ekonify Retrain API v2.0 is running!"}

# Add the router to the FastAPI app
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("retrain:app", host="127.0.0.1", port=8000, reload=True)
