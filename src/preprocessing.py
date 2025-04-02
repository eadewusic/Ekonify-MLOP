import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
from glob import glob
import zipfile
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def extract_dataset(zip_path, extract_path):
    """Extract dataset from zip file"""
    if not os.path.exists(zip_path):
        print(f"Error: Dataset zip file not found at {zip_path}")
        return False
    
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(extract_path)
    zip_ref.close()
    print("Dataset extracted successfully!")
    return True

def display_sample_image(dataset_path):
    """Display a sample image from the dataset"""
    # Find image files inside all subfolders
    image_files = glob(os.path.join(dataset_path, "*", "*.jpg")) + \
                  glob(os.path.join(dataset_path, "*", "*.png")) + \
                  glob(os.path.join(dataset_path, "*", "*.jpeg"))

    if not image_files:
        print("No images found in subfolders.")
        return False
    else:
        print(f"Found {len(image_files)} images.")

    # Use the first image
    first_image_path = image_files[0]
    print("First image path:", first_image_path)

    # Load the first image
    image = cv2.imread(first_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Show the image
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    return True

def copy_dataset(src, dst):
    """Copy dataset from source to destination"""
    if os.path.exists(dst):
        print("Copy already exists. Skipping copying step.")
        return
    
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"Copied dataset to {dst}")

def show_class_distribution(data_dir):
    """Show class distribution in the dataset"""
    # Ensure we only count directories (not files)
    class_counts = {
        cls: len(os.listdir(os.path.join(data_dir, cls)))
        for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))
    }

    # Plot class distribution
    plt.figure(figsize=(8, 5))
    plt.bar(class_counts.keys(), class_counts.values(), color=['blue', 'black'])
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution")
    plt.xticks(rotation=20)
    plt.show()

    print("Class counts:", class_counts)
    
    return class_counts

def oversample_data(copied_data_dir, new_data_dir):
    """Oversample smaller classes to balance the dataset"""
    # Check if copied_data_dir exists
    if not os.path.exists(copied_data_dir):
        print(f"Error: The source directory '{copied_data_dir}' does not exist.")
        return

    # Get the class counts from the source directory
    class_counts = {
        cls: len(os.listdir(os.path.join(copied_data_dir, cls))) 
        for cls in os.listdir(copied_data_dir) if os.path.isdir(os.path.join(copied_data_dir, cls))
    }
    print(f"Class counts in source directory: {class_counts}")

    # Find the class with the most images (target for oversampling)
    max_class = max(class_counts, key=class_counts.get)
    max_count = class_counts[max_class]

    # Create new data directory if it doesn't exist
    if not os.path.exists(new_data_dir):
        os.makedirs(new_data_dir)

    # Loop through each class in the copied data directory
    for cls in os.listdir(copied_data_dir):
        cls_path = os.path.join(copied_data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
            
        new_class_path = os.path.join(new_data_dir, cls)

        # Create class folder in the new directory if it doesn't exist
        if not os.path.exists(new_class_path):
            os.makedirs(new_class_path)

        # Get all image paths for the current class
        images = os.listdir(cls_path)
        num_images = len(images)

        print(f"Processing class '{cls}': {num_images} images found.")

        # Always copy original images first
        for img in images:
            img_path = os.path.join(cls_path, img)
            if os.path.isfile(img_path):
                shutil.copy(img_path, os.path.join(new_class_path, img))

        # If the current class has fewer images than the max, duplicate images
        if num_images < max_count:
            additional_needed = max_count - num_images
            print(f"Class '{cls}' needs {additional_needed} more images to match {max_count}. Duplicating...")
            
            for i in range(additional_needed):
                img_to_copy = random.choice(images)
                original_path = os.path.join(cls_path, img_to_copy)
                
                if os.path.isfile(original_path):
                    # Create a new filename to avoid overwriting existing files
                    base_name, ext = os.path.splitext(img_to_copy)
                    new_name = f"{base_name}_dup{i}{ext}"
                    shutil.copy(original_path, os.path.join(new_class_path, new_name))

        print(f"Oversampled {cls} to {max_count} images in new dataset.")

def split_dataset(source_dir, train_dir, val_dir, test_dir, split=(0.8, 0.1, 0.1)):
    """Split dataset into training, validation, and test sets"""
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)

        if os.path.isdir(class_path):
            # Create class directories in train, val, and test
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            # Get all image filenames
            filenames = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            random.shuffle(filenames)

            # Calculate the split indices
            total_files = len(filenames)
            train_end = int(total_files * split[0])
            val_end = int(total_files * (split[0] + split[1]))

            # Split the filenames
            train_files = filenames[:train_end]
            val_files = filenames[train_end:val_end]
            test_files = filenames[val_end:]

            # Copy files to corresponding directories (instead of moving)
            for file in train_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(train_dir, class_name, file))
            for file in val_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(val_dir, class_name, file))
            for file in test_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(test_dir, class_name, file))

            # Print out confirmation for each class
            print(f"Class '{class_name}' has been split and saved into train, val, and test directories.")

def create_data_generators(train_dir, val_dir, test_dir, img_height=224, img_width=224, batch_size=32):
    """Create data generators for training, validation, and testing"""
    # Define ImageDataGenerator for the training, validation, and test sets
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,  # Rescale pixel values to [0, 1]
        rotation_range=40,  # Randomly rotate images by up to 40 degrees
        width_shift_range=0.2,  # Randomly shift images horizontally by 20%
        height_shift_range=0.2,  # Randomly shift images vertically by 20%
        shear_range=0.2,  # Apply random shear transformations
        zoom_range=0.2,  # Apply random zoom transformations
        horizontal_flip=True,  # Randomly flip images horizontally
        fill_mode='nearest'  # Fill in missing pixels with the nearest valid pixel
    )

    val_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Only rescale for validation
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Only rescale for testing

    # Create generators for train, validation, and test data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),  # Resize to the input shape your model expects
        batch_size=batch_size,
        class_mode='categorical'  # For multi-class classification
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, validation_generator, test_generator

def main():
    # Define paths
    # You can change these paths to match your local environment
    dataset_zip_path = "./garbage_images.zip"
    extract_path = "./garbage_classification"
    copied_data_dir = "./copied_dataset"
    oversampled_data_dir = "./oversampled_dataset"
    train_dir = "./train_data"
    val_dir = "./val_data"
    test_dir = "./test_data"
    
    # Extract dataset
    if not os.path.exists(extract_path):
        extract_dataset(dataset_zip_path, ".")
    
    # Display a sample image
    display_sample_image(extract_path)
    
    # Copy dataset
    copy_dataset(extract_path, copied_data_dir)
    
    # Show class distribution
    show_class_distribution(copied_data_dir)
    
    # Oversample data
    oversample_data(copied_data_dir, oversampled_data_dir)
    
    # Show class distribution after oversampling
    show_class_distribution(oversampled_data_dir)
    
    # Split dataset
    split_dataset(oversampled_data_dir, train_dir, val_dir, test_dir)
    
    # Create data generators
    train_generator, validation_generator, test_generator = create_data_generators(
        train_dir, val_dir, test_dir
    )
    
    print("Preprocessing completed successfully!")
    print(f"Train data directory: {train_dir}")
    print(f"Validation data directory: {val_dir}")
    print(f"Test data directory: {test_dir}")
    
    # Get class names
    class_names = list(train_generator.class_indices.keys())
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")

if __name__ == "__main__":
    main()
