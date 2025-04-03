# My baseline model is 127 MB which exceeds GitHub storage and LFS of 100 MB
# Hence using Google Drive approach

import gdown
import os

# Google Drive File ID
file_id = "1mM_9Su-I7iXqJWSRMQ1idK2zxTji6ZVT"

# Define output path
save_dir = "models"
output = os.path.join(save_dir, "baseline_cnn.keras")

# Ensure the models directory exists
os.makedirs(save_dir, exist_ok=True)

# Check if the model is already downloaded
if os.path.exists(output):
    print("Model already exists. Skipping download.")
else:
    print("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    print("Download complete!")
