import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, Input
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

def load_baseline_model(model_path):
    """Load a pre-trained baseline model"""
    try:
        baseline_model = load_model(model_path)
        print("Baseline Model loaded successfully!")
        baseline_model.summary()
        return baseline_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_improved_model(baseline_model):
    """Create an improved model based on the baseline model"""
    # Use input from first real layer
    input_layer = baseline_model.layers[0].input
    x = input_layer

    conv2d_count = 0

    # Loop through *all* layers, including the first
    for layer in baseline_model.layers:
        config = layer.get_config()

        # Only get weights if NOT Flatten or Dense
        if not isinstance(layer, (layers.Flatten, layers.Dense)):
            weights = layer.get_weights()  # Preserve the original layer's weights

        if isinstance(layer, layers.Conv2D):
            conv2d_count += 1
            config['kernel_regularizer'] = regularizers.l2(0.00004)
            new_layer = layers.Conv2D.from_config(config)  # Create the layer
            new_layer.build(input_shape=x.shape)  # Build the layer if needed

            # Set weights only if they were obtained
            if not isinstance(layer, (layers.Flatten, layers.Dense)):
                new_layer.set_weights(weights)  # Copy weights from original layer

            x = new_layer(x)  # Apply the layer

        elif isinstance(layer, layers.MaxPooling2D):
            x = layers.MaxPooling2D.from_config(config)(x)
            x = layers.Dropout(0.3)(x)

        elif isinstance(layer, layers.Flatten):
            x = layers.GlobalAveragePooling2D()(x)

        elif isinstance(layer, layers.Dense):
            config['kernel_regularizer'] = regularizers.l2(0.000004)

            # Create the Dense layer without applying it immediately
            new_layer = layers.Dense.from_config(config)

            # Build the layer if needed
            new_layer.build(input_shape=x.shape)

            # Now apply the layer
            x = new_layer(x)

            # Dropout only after first Dense layer
            if layer.name != 'dense_1':  # Assuming 'dense_1' is the first Dense layer
                x = layers.Dropout(0.4)(x)

        else:
            new_layer = layer.__class__.from_config(config)(x)

            # Set weights only if they were obtained
            if not isinstance(layer, (layers.Flatten, layers.Dense)):
                new_layer.set_weights(weights)  # Copy weights from original layer

            x = new_layer

    # Check Conv2D layer count
    if conv2d_count < 3:
        print(f"Warning: Only {conv2d_count} Conv2D layers detected — you may need to verify layer[0].")
    else:
        print(f"Model includes all {conv2d_count} Conv2D layers.")

    # Build new model
    improved_model = models.Model(inputs=input_layer, outputs=x)

    # Compile with your desired learning rate
    improved_model.compile(
        optimizer=AdamW(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    improved_model.summary()
    return improved_model

def train_model(model, train_generator, validation_generator, model_save_path, epochs=20):
    """Train the model"""
    # Early stopping to stop training when validation loss is not improving
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Save the model with the best validation accuracy
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint]
    )
    
    return history

def plot_history(history, model_name="Retrained Model"):
    """Plot training history"""
    plt.figure(figsize=(12, 5))

    # Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training & Validation Loss')
    plt.legend()

    # Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Training & Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_generator):
    """Evaluate model on test data"""
    # Evaluate model on test data
    test_performance = model.evaluate(test_generator)
    print(f"Model - Test Loss: {test_performance[0]}, Test Accuracy: {test_performance[1]}")
    
    # Generate predictions for the test set
    test_generator.reset()  # Ensure the generator starts from the beginning
    predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)

    # Get the predicted class indices
    y_pred_classes = np.argmax(predictions, axis=1)

    # Get the true labels from the generator
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # Print confusion matrix as text
    print("Confusion Matrix (as text):")
    print(cm)

    # Visualize Confusion Matrix using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Classification Report
    report = classification_report(y_true, y_pred_classes, target_names=class_labels)
    print("Classification Report:\n", report)

    # Calculate Accuracy, Precision, Recall, F1 Score
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')

    # Print Performance Metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'loss': test_performance[0],
        'accuracy': test_performance[1],
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

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
    train_dir = "./train_data"
    val_dir = "./val_data"
    test_dir = "./test_data"
    baseline_model_path = "./saved_models/baseline_cnn.keras"
    improved_model_save_path = "./retrained_baseline_model.keras"
    
    # Create data generators
    train_generator, validation_generator, test_generator = create_data_generators(
        train_dir, val_dir, test_dir
    )
    
    # Load baseline model
    baseline_model = load_baseline_model(baseline_model_path)
    
    if baseline_model is None:
        print("Cannot proceed without a baseline model.")
        return
    
    # Create improved model
    improved_model = create_improved_model(baseline_model)
    
    # Train model
    history = train_model(
        improved_model, 
        train_generator, 
        validation_generator, 
        improved_model_save_path
    )
    
    # Plot training history
    plot_history(history, "Retrained Baseline CNN")
    
    # Evaluate model
    metrics = evaluate_model(improved_model, test_generator)
    
    print("Model training and evaluation completed successfully!")
    print(f"Model saved to: {improved_model_save_path}")
    print(f"Final model accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
