"""
Lionfish Detection API
Flask API that serves the trained model and provides endpoints to classify images

Usage:
    python lionfish_api.py

API Endpoints:
    GET /                    - Web interface / API info
    GET /health              - Health check
    GET /classify/random     - Classify a random image
    GET /classify/random/image - Get the classified image
    GET /stats               - Dataset and model statistics
"""

from flask import Flask, jsonify, send_file
import tensorflow as tf
import numpy as np
import os
import random
from PIL import Image
import io

app = Flask(__name__)

# =============================================================================
# CONFIGURATION - Update these paths for your system
# =============================================================================

# Path to the trained model file (.h5 or .keras)
MODEL_PATH = os.environ.get('MODEL_PATH', './model/lionfish_model.h5')

# Path to the dataset directory (containing train/val/test folders)
DATASET_PATH = os.environ.get('DATASET_PATH', './dataset')

# =============================================================================

model = None


def load_model():
    """Load the trained model"""
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print(f"✗ Model not found at {MODEL_PATH}")
        print("Please download the model or train your own!")


def get_all_images():
    """Get list of all images from the dataset"""
    images = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(DATASET_PATH, split)
        for class_name in ['lionfish', 'not lionfish']:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images.append({
                            'path': os.path.join(class_dir, img_file),
                            'true_label': class_name,
                            'split': split
                        })
    return images


def preprocess_image_for_prediction(image_path):
    """Preprocess an image for model prediction"""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = img / 255.0
    img = tf.expand_dims(img, 0)  # Add batch dimension
    return img


@app.route('/')
def home():
    """API home endpoint - serves the HTML viewer if available"""
    # Try to find the viewer HTML file
    possible_paths = [
        './web/viewer.html',
        '../web/viewer.html',
        './viewer.html'
    ]
    
    for html_path in possible_paths:
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    # Return JSON API info if no HTML found
    return jsonify({
        'message': 'Lionfish Detection API',
        'version': '1.0',
        'endpoints': {
            '/': 'API information',
            '/health': 'Check API status',
            '/classify/random': 'Classify a random image from dataset',
            '/classify/random/image': 'Get the random image being classified',
            '/stats': 'Get model statistics'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'dataset_found': os.path.exists(DATASET_PATH)
    })


@app.route('/classify/random')
def classify_random():
    """Classify a random image from the dataset"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Get all images
    all_images = get_all_images()

    if not all_images:
        return jsonify({'error': 'No images found in dataset'}), 404

    # Select random image
    selected = random.choice(all_images)

    # Preprocess and predict
    img_tensor = preprocess_image_for_prediction(selected['path'])
    prediction = model.predict(img_tensor, verbose=0)

    # Interpret prediction
    # Model outputs: 0 = lionfish, 1 = not lionfish
    confidence = float(prediction[0][0])
    predicted_class = 'Not Lionfish' if confidence > 0.5 else 'Lionfish'

    # Store the last selected image path for the image endpoint
    with open('last_image.txt', 'w') as f:
        f.write(selected['path'])

    # Calculate correct confidence percentage
    if predicted_class == 'Lionfish':
        confidence_percent = f"{(1-confidence) * 100:.2f}%"
    else:
        confidence_percent = f"{confidence * 100:.2f}%"

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': confidence,
        'confidence_percent': confidence_percent,
        'true_label': selected['true_label'],
        'correct': (predicted_class.lower() == selected['true_label'].lower()),
        'split': selected['split'],
        'image_filename': os.path.basename(selected['path']),
        'image_url': '/classify/random/image'
    })


@app.route('/classify/random/image')
def get_random_image():
    """Get the last randomly selected image"""
    if not os.path.exists('last_image.txt'):
        return jsonify({'error': 'No image selected yet. Call /classify/random first'}), 404

    with open('last_image.txt', 'r') as f:
        image_path = f.read().strip()

    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404

    # Open and return the image
    img = Image.open(image_path)
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


@app.route('/stats')
def stats():
    """Get dataset and model statistics"""
    all_images = get_all_images()

    stats = {
        'total_images': len(all_images),
        'by_split': {},
        'by_class': {}
    }

    for img in all_images:
        # Count by split
        if img['split'] not in stats['by_split']:
            stats['by_split'][img['split']] = 0
        stats['by_split'][img['split']] += 1

        # Count by class
        if img['true_label'] not in stats['by_class']:
            stats['by_class'][img['true_label']] = 0
        stats['by_class'][img['true_label']] += 1

    if model is not None:
        stats['model'] = {
            'parameters': int(model.count_params()),
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape)
        }

    return jsonify(stats)


if __name__ == '__main__':
    print("=" * 50)
    print("  Lionfish Detection API")
    print("=" * 50)
    print(f"\nModel path: {MODEL_PATH}")
    print(f"Dataset path: {DATASET_PATH}\n")
    
    load_model()
    
    print("\nAPI Endpoints:")
    print("  http://localhost:5000/")
    print("  http://localhost:5000/classify/random")
    print("  http://localhost:5000/classify/random/image")
    print("  http://localhost:5000/stats")
    print("\nStarting server...\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
