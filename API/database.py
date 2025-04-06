from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from pymongo.errors import ConnectionFailure, PyMongoError


class Database:
    def __init__(self, db_name="Ekonify", uri="mongodb://localhost:27017"):
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')  # Checking the connection
            print("Connected to MongoDB successfully!")
        except ConnectionFailure:
            print("Failed to connect to MongoDB server.")
            raise
        
        self.db = self.client[db_name]
        self.images = self.db["images"]
        self.retraining = self.db["retraining"]

    def insert_image(self, image_data, label, metadata):
        try:
            document = {
                "image_data": image_data,
                "label": label,
                "timestamp": datetime.now(),
                "metadata": metadata
            }
            result = self.images.insert_one(document)
            return str(result.inserted_id)
        except PyMongoError as e:
            print(f"Error inserting image: {e}")
            return None

    def retrieve_image(self, image_id):
        try:
            result = self.images.find_one({"_id": ObjectId(image_id)})
            return result if result else None
        except PyMongoError as e:
            print(f"Error retrieving image: {e}")
            return None

    def update_image(self, image_id, updated_data):
        try:
            result = self.images.update_one({"_id": ObjectId(image_id)}, {"$set": updated_data})
            return result.modified_count > 0
        except PyMongoError as e:
            print(f"Error updating image: {e}")
            return False

    def delete_image(self, image_id):
        try:
            result = self.images.delete_one({"_id": ObjectId(image_id)})
            return result.deleted_count > 0
        except PyMongoError as e:
            print(f"Error deleting image: {e}")
            return False

    def insert_for_retraining(self, image_data, label, feedback):
        try:
            document = {
                "image_data": image_data,
                "label": label,
                "timestamp": datetime.now(),
                "feedback": feedback
            }
            result = self.retraining.insert_one(document)
            return str(result.inserted_id)
        except PyMongoError as e:
            print(f"Error inserting retraining image: {e}")
            return None

    def retrieve_for_retraining(self):
        try:
            result = list(self.retraining.find())
            return result
        except PyMongoError as e:
            print(f"Error retrieving retraining images: {e}")
            return []

    def delete_retraining_image(self, image_id):
        try:
            result = self.retraining.delete_one({"_id": ObjectId(image_id)})
            return result.deleted_count > 0
        except PyMongoError as e:
            print(f"Error deleting retraining image: {e}")
            return False

    def __del__(self):
        self.client.close()


if __name__ == "__main__":
    db = Database()
