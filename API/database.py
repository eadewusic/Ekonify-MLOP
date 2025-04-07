from pymongo import MongoClient
from gridfs import GridFS
from bson import ObjectId
from datetime import datetime
from pymongo.errors import ConnectionFailure, PyMongoError
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_name="Ekonify", uri="mongodb://adminUser:Climiradi_01@127.0.0.1:27017/?authSource=admin"):
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')  # Test connection
            logger.info("Connected to MongoDB successfully!")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        
        self.db = self.client[db_name]
        self.fs = GridFS(self.db)  # For large file storage
        self.images = self.db["images"]
        self.retraining = self.db["retraining"]
        self.training_history = self.db["training_history"]

    # Image operations
    def insert_image(self, file_data, filename, label, metadata=None):
        """Store an image in GridFS and its metadata in images collection"""
        try:
            if metadata is None:
                metadata = {}
            
            # Store file in GridFS
            file_id = self.fs.put(file_data, filename=filename)
            
            # Store metadata
            document = {
                "file_id": file_id,
                "label": label,
                "filename": filename,
                "timestamp": datetime.now(),
                "used_in_training": False,  # Track if this image has been used in training
                "metadata": metadata
            }
            result = self.images.insert_one(document)
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Error inserting image: {e}")
            return None

    def insert_many_images(self, image_data_list):
        """Bulk insert images with their metadata"""
        try:
            results = []
            for img_data, filename, label, metadata in image_data_list:
                results.append(self.insert_image(img_data, filename, label, metadata))
            return results
        except PyMongoError as e:
            logger.error(f"Error bulk inserting images: {e}")
            return None

    def get_image(self, image_id):
        """Retrieve an image and its metadata"""
        try:
            doc = self.images.find_one({"_id": ObjectId(image_id)})
            if not doc:
                return None
                
            file_data = self.fs.get(doc["file_id"]).read()
            return {
                "file_data": file_data,
                "metadata": doc
            }
        except PyMongoError as e:
            logger.error(f"Error retrieving image: {e}")
            return None

    # Retraining operations
    def add_to_retraining(self, image_id, feedback=None):
        """Add an existing image to retraining set"""
        try:
            doc = self.images.find_one({"_id": ObjectId(image_id)})
            if not doc:
                return False
                
            retrain_doc = {
                "image_id": image_id,
                "label": doc["label"],
                "timestamp": datetime.now(),
                "feedback": feedback or ""
            }
            self.retraining.insert_one(retrain_doc)
            return True
        except PyMongoError as e:
            logger.error(f"Error adding to retraining: {e}")
            return False

    def get_retraining_batch(self, limit=100):
        """Get a batch of images for retraining"""
        try:
            retrain_docs = list(self.retraining.find().limit(limit))
            images = []
            
            for doc in retrain_docs:
                img = self.get_image(doc["image_id"])
                if img:
                    images.append({
                        "data": img["file_data"],
                        "label": doc["label"],
                        "metadata": img["metadata"]
                    })
            return images
        except PyMongoError as e:
            logger.error(f"Error getting retraining batch: {e}")
            return []

    def get_all_training_images(self, limit=1000):
        """Get all images from the database for training, prioritizing ones not used in training yet"""
        try:
            # First get images that haven't been used in training
            image_docs = list(self.images.find({"used_in_training": False}).limit(limit))
            
            # If we don't have enough, get some that have been used before
            if len(image_docs) < limit:
                remaining = limit - len(image_docs)
                used_images = list(self.images.find({"used_in_training": True}).limit(remaining))
                image_docs.extend(used_images)
            
            images = []
            for doc in image_docs:
                try:
                    file_data = self.fs.get(doc["file_id"]).read()
                    images.append({
                        "data": file_data,
                        "label": doc["label"],
                        "_id": str(doc["_id"])
                    })
                except Exception as e:
                    logger.error(f"Error retrieving file {doc['file_id']}: {e}")
                    continue
                    
            logger.info(f"Retrieved {len(images)} images for training")
            return images
        except PyMongoError as e:
            logger.error(f"Error retrieving training images: {e}")
            return []

    def mark_images_as_trained(self, image_ids=None):
        """Mark images as used in training
        
        Args:
            image_ids: List of image IDs. If None, marks all images.
        """
        try:
            query = {"_id": {"$in": [ObjectId(id) for id in image_ids]}} if image_ids else {}
            result = self.images.update_many(
                query, 
                {"$set": {"used_in_training": True, "last_trained_date": datetime.now()}}
            )
            logger.info(f"Marked {result.modified_count} images as used in training")
            return result.modified_count
        except PyMongoError as e:
            logger.error(f"Error marking images as trained: {e}")
            return 0

    def record_training_session(self, model_name, metrics, image_ids):
        """Record a training session in the database"""
        try:
            doc = {
                "model_name": model_name,
                "timestamp": datetime.now(),
                "metrics": metrics,
                "image_count": len(image_ids),
                "image_ids": image_ids
            }
            result = self.training_history.insert_one(doc)
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Error recording training session: {e}")
            return None

    def clear_retraining_batch(self):
        """Clear processed retraining images"""
        try:
            result = self.retraining.delete_many({})
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Error clearing retraining batch: {e}")
            return 0
            
    # MODEL STORAGE AND RETRIEVAL METHODS USING GRIDFS
    
    def store_model(self, model_binary, model_name, base_model_name=None, metrics=None):
        """Store a trained model in the database using GridFS for the large binary data"""
        try:
            # Store the large model binary in GridFS
            model_file_id = self.fs.put(
                model_binary, 
                filename=f"{model_name}.pkl",
                contentType="application/octet-stream"
            )
            
            # Create a new collection for models if it doesn't exist
            if "models" not in self.db.list_collection_names():
                self.db.create_collection("models")
            
            # Store metadata in the models collection
            model_doc = {
                "model_name": model_name,
                "base_model_name": base_model_name,
                "model_file_id": model_file_id,  # Reference to GridFS instead of embedding binary
                "file_size_bytes": len(model_binary),
                "timestamp": datetime.now(),
                "metrics": metrics or {}
            }
            
            result = self.db.models.insert_one(model_doc)
            logger.info(f"Model {model_name} stored in database with ID {result.inserted_id}")
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Error saving model to database: {e}")
            raise Exception(f"Error saving model to database: {e}")
            
    def get_latest_retrained_model(self, base_model_name):
        """Get the latest retrained model for a given base model"""
        try:
            # Check if models collection exists
            if "models" not in self.db.list_collection_names():
                logger.warning("Models collection does not exist in database")
                return None
            
            # Find the most recent model with the matching base name
            model_doc = self.db.models.find_one(
                {"base_model_name": base_model_name},
                sort=[("timestamp", -1)]  # Sort by timestamp descending (most recent first)
            )
            
            if not model_doc:
                logger.warning(f"No retrained model found for base model {base_model_name}")
                return None
            
            # Retrieve the model binary from GridFS
            if "model_file_id" in model_doc:
                model_binary = self.fs.get(model_doc["model_file_id"]).read()
                model = pickle.loads(model_binary)
                logger.info(f"Retrieved latest retrained model: {model_doc['model_name']}")
                return model, model_doc
            else:
                logger.error(f"Model document missing GridFS file_id reference: {model_doc['_id']}")
                return None
        except PyMongoError as e:
            logger.error(f"Error retrieving model from database: {e}")
            return None
            
    def list_available_models(self, base_model_name=None):
        """List all available models or those derived from a specific base model"""
        try:
            # Check if models collection exists
            if "models" not in self.db.list_collection_names():
                return []
            
            # Set up query to filter by base model name if provided
            query = {"base_model_name": base_model_name} if base_model_name else {}
            
            # Return model information without the binary data
            models = list(self.db.models.find(query))
            
            # Convert ObjectId to string
            for model in models:
                model["_id"] = str(model["_id"])
                model["model_file_id"] = str(model["model_file_id"])
            
            return models
        except PyMongoError as e:
            logger.error(f"Error listing models: {e}")
            return []
            
    def get_model_by_id(self, model_id):
        """Retrieve a specific model by its ID"""
        try:
            # Check if models collection exists
            if "models" not in self.db.list_collection_names():
                logger.warning("Models collection does not exist in database")
                return None
                
            # Find the model by ID
            model_doc = self.db.models.find_one({"_id": ObjectId(model_id)})
            
            if not model_doc:
                logger.warning(f"No model found with ID: {model_id}")
                return None
                
            # Retrieve the model binary from GridFS
            if "model_file_id" in model_doc:
                model_binary = self.fs.get(model_doc["model_file_id"]).read()
                model = pickle.loads(model_binary)
                logger.info(f"Retrieved model: {model_doc['model_name']}")
                return model, model_doc
            else:
                logger.error(f"Model document missing GridFS file_id reference: {model_doc['_id']}")
                return None
        except PyMongoError as e:
            logger.error(f"Error retrieving model by ID: {e}")
            return None
            
    def store_dataset(self, zip_content):
        """Store a dataset ZIP file in the database"""
        try:
            # Store the dataset in GridFS
            file_id = self.fs.put(
                zip_content, 
                filename=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                contentType="application/zip"
            )
            
            # Create datasets collection if it doesn't exist
            if "datasets" not in self.db.list_collection_names():
                self.db.create_collection("datasets")
                
            # Store metadata in database
            dataset_doc = {
                "timestamp": datetime.now(),
                "dataset_file_id": file_id,  # Reference to GridFS file
                "size_bytes": len(zip_content)
            }
            
            result = self.db.datasets.insert_one(dataset_doc)
            logger.info(f"Dataset stored in database with ID {result.inserted_id}")
            return str(result.inserted_id)
        except PyMongoError as e:
            logger.error(f"Error storing dataset: {e}")
            raise Exception(f"Error storing dataset: {e}")
            
    def get_all_datasets(self, limit=10):
        """Get all stored datasets, limited to the most recent ones"""
        try:
            # Check if datasets collection exists
            if "datasets" not in self.db.list_collection_names():
                logger.warning("Datasets collection does not exist in database")
                return []
                
            # Get the most recent datasets
            datasets = list(self.db.datasets.find().sort("timestamp", -1).limit(limit))
            
            # Fetch the actual zip data from GridFS for each dataset
            for dataset in datasets:
                if "dataset_file_id" in dataset:
                    dataset["zip_data"] = self.fs.get(dataset["dataset_file_id"]).read()
                    dataset["_id"] = str(dataset["_id"])
                    dataset["dataset_file_id"] = str(dataset["dataset_file_id"])
                else:
                    logger.warning(f"Dataset missing GridFS file_id reference: {dataset.get('_id')}")
                
            logger.info(f"Retrieved {len(datasets)} datasets")
            return datasets
        except PyMongoError as e:
            logger.error(f"Error retrieving datasets: {e}")
            return []

    def __del__(self):
        try:
            self.client.close()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")

if __name__ == "__main__":
    # Test connection
    db = Database()
    print("Database connection test successful!")
