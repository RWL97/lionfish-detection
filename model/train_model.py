"""
Lionfish Detection Model Training Script

This script trains a CNN model to classify images as either 
containing a lionfish or not (binary classification).

Dataset Structure:
    dataset/
    ├── train/
    │   ├── lionfish/
    │   └── not lionfish/
    ├── val/
    │   ├── lionfish/
    │   └── not lionfish/
    └── test/
        ├── lionfish/
        └── not lionfish/

Usage:
    python train_model.py
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# =============================================================================
# CONFIGURATION
# =============================================================================

# Set a random seed for reproducibility
SEED_VALUE = 97
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# Dataset paths - UPDATE THESE FOR YOUR SYSTEM
BASE_DIR = os.environ.get('DATASET_PATH', './dataset')
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# Model save path
MODEL_SAVE_PATH = './lionfish_model.h5'

# Image dimensions and batch size
IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 32

# Training parameters
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 3

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def preprocess_image(file_path):
    """Normalize and resize an image for the model"""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0  # Normalize to [0, 1]
    return img


def load_and_preprocess_image(file_path, label):
    """Wrapper to return both processed image and label"""
    return preprocess_image(file_path), label


def create_dataset(directory, batch_size, shuffle=True, repeat=True):
    """
    Create a TensorFlow dataset from image files in a directory.
    
    Args:
        directory: Path to directory containing class subdirectories
        batch_size: Number of images per batch
        shuffle: Whether to shuffle the dataset
        repeat: Whether to repeat the dataset indefinitely
    
    Returns:
        dataset: TensorFlow dataset
        total_samples: Number of images in the dataset
    """
    # Get class labels from directory names
    labels = sorted([d for d in os.listdir(directory) 
                    if os.path.isdir(os.path.join(directory, d))])
    label_to_index = {label: index for index, label in enumerate(labels)}
    
    print(f"Classes found: {labels}")
    
    file_paths = []
    file_labels = []
    
    for label in labels:
        class_dir = os.path.join(directory, label)
        class_files = [os.path.join(class_dir, fname) 
                      for fname in os.listdir(class_dir)
                      if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
        file_paths.extend(class_files)
        file_labels.extend([label_to_index[label]] * len(class_files))
        print(f"  {label}: {len(class_files)} images")
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, file_labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths), seed=SEED_VALUE)
    
    if repeat:
        dataset = dataset.repeat()
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset, len(file_paths)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def build_model():
    """
    Build the CNN model for binary classification.
    
    Architecture:
        - 3 Convolutional layers with increasing filters (32 -> 64 -> 128)
        - MaxPooling after each conv layer
        - Dense layer with dropout for regularization
        - Sigmoid output for binary classification
    """
    model = tf.keras.models.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        # First conv block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second conv block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Third conv block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        
        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =============================================================================
# TRAINING
# =============================================================================

def train_model():
    """Main training function"""
    
    print("=" * 60)
    print("  Lionfish Detection Model Training")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(TRAIN_DIR):
        print(f"\nError: Training directory not found at {TRAIN_DIR}")
        print("Please update BASE_DIR in the script or set DATASET_PATH environment variable.")
        return
    
    # Create datasets
    print("\n📁 Loading datasets...")
    train_dataset, train_size = create_dataset(TRAIN_DIR, BATCH_SIZE)
    val_dataset, val_size = create_dataset(VAL_DIR, BATCH_SIZE, shuffle=False)
    test_dataset, test_size = create_dataset(TEST_DIR, BATCH_SIZE, shuffle=False, repeat=False)
    
    print(f"\nTotal: {train_size} training, {val_size} validation, {test_size} test images")
    
    # Calculate steps
    steps_per_epoch = np.ceil(train_size / BATCH_SIZE).astype(int)
    validation_steps = np.ceil(val_size / BATCH_SIZE).astype(int)
    test_steps = np.ceil(test_size / BATCH_SIZE).astype(int)
    
    # Calculate class weights for imbalanced data
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique([0, 1]),
        y=np.concatenate([np.zeros(train_size // 2), np.ones(train_size // 2)])
    )
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    
    # Build model
    print("\n🏗️ Building model...")
    model = build_model()
    model.summary()
    
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )
    
    # Train the model
    print("\n🚀 Starting training...")
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=[early_stopping]
    )
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_dataset, steps=test_steps)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    test_predictions = model.predict(test_dataset, steps=test_steps, verbose=1)
    test_predictions = np.where(test_predictions > 0.5, 1, 0)
    
    # Get true labels
    true_labels = np.concatenate([y for x, y in test_dataset], axis=0)
    test_predictions = test_predictions[:len(true_labels)]
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(true_labels, test_predictions, 
                               target_names=['Not Lionfish', 'Lionfish']))
    
    # Print confusion matrix
    cm = confusion_matrix(true_labels, test_predictions)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate metrics
    precision = precision_score(true_labels, test_predictions, zero_division=1)
    recall = recall_score(true_labels, test_predictions, zero_division=1)
    f1 = f1_score(true_labels, test_predictions, zero_division=1)
    
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save the model
    print(f"\n💾 Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully!")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history


def plot_training_history(history):
    """Plot training and validation accuracy/loss curves"""
    plt.figure(figsize=(14, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("\n📈 Training plots saved to training_history.png")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    train_model()
