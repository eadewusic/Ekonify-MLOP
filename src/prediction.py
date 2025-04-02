import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import Image

def load_trained_model(model_path):
    """Load a trained model"""
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_class_names(test_dir):
    """Get class names from the test directory"""
    # Create a temporary ImageDataGenerator to get class names
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    
    class_names = list(test_generator.class_indices.keys())
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    
    return class_names, test_generator

def predict_single_image(model, image_data, class_names):
    """Predict class for a single image"""
    # Convert image to array and expand dims for prediction
    img_array = tf.keras.preprocessing.image.img_to_array(image_data)
    img_array = img_array / 255.0  # Normalize
    img_array = tf.expand_dims(img_array, 0)  # Model expects batch format

    # Make prediction
    predictions = model.predict(img_array)
    
    # Get predicted class and confidence
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence_level = f"{100 * np.max(predictions[0]):.2f}"
    
    return predicted_class, confidence_level, predictions[0]

def show_sample_predictions(model, test_generator, class_names, num_samples=3):
    """Show predictions for sample images from test generator"""
    # Get a single batch of data from the test_generator
    images_batch, labels_batch = next(test_generator)

    plt.figure(figsize=(12, 4 * num_samples))
    for i in range(num_samples):  # Predict for num_samples images from the batch
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(images_batch[i])

        predicted_class, confidence_level, _ = predict_single_image(
            model, images_batch[i], class_names
        )
        actual_class = class_names[np.argmax(labels_batch[i])]  # Ensure correct label index

        plt.title(f'Actual: {actual_class}\nPredicted: {predicted_class}\nConfidence: {confidence_level}%')
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def predict_from_file(model, image_path, class_names):
    """Predict class for an image file"""
    try:
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(224, 224)
        )
        
        # Display the image
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        
        # Make prediction
        predicted_class, confidence_level, predictions = predict_single_image(
            model, img, class_names
        )
        
        plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence_level}%')
        plt.show()
        
        # Print top 3 predictions
        print("\nTop 3 Predictions:")
        indices = np.argsort(predictions)[::-1][:3]
        for i, idx in enumerate(indices):
            print(f"{i+1}. {class_names[idx]} - {predictions[idx]*100:.2f}%")
            
        return predicted_class, confidence_level, predictions
        
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None, None

def evaluate_on_test_set(model, test_generator, class_names):
    """Evaluate model on test set and show confusion matrix"""
    # Get predictions
    test_generator.reset()
    predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Calculate and show metrics
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def main():
    # Define paths
    # You can change these paths to match your local environment
    model_path = "./retrained_baseline_model.keras"
    test_dir = "./test_data"
    
    # Load model
    model = load_trained_model(model_path)
    
    if model is None:
        print("Cannot proceed without a model.")
        return
    
    # Get class names and test generator
    class_names, test_generator = get_class_names(test_dir)
    
    # Show sample predictions
    show_sample_predictions(model, test_generator, class_names)
    
    # Evaluate on test set
    evaluate_on_test_set(model, test_generator, class_names)
    
    # Predict from file (optional)
    # If you want to predict from a file, uncomment the following lines
    # image_path = "./test_image.jpg"  # Path to your test image
    # if os.path.exists(image_path):
    #     predict_from_file(model, image_path, class_names)
    # else:
    #     print(f"Test image not found at {image_path}")
    
    print("\nPrediction functionality ready!")
    print("You can use this script to make predictions on new images.")
    print("To predict on a new image, use the predict_from_file function.")

if __name__ == "__main__":
    main()
