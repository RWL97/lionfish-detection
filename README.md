# 🐟 Lionfish Detection Model

A deep learning image classification model that detects invasive lionfish in underwater images using TensorFlow/Keras CNN.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> This project was developed as a Master's thesis. Read the full paper: [thesis.pdf](docs/thesis.pdf)

## Project Overview

Lionfish are an invasive species that pose a significant threat to marine ecosystems, particularly in the Atlantic Ocean and Caribbean Sea. This project uses a Convolutional Neural Network (CNN) to classify underwater images as either containing a lionfish or not.

### Features
- **Binary Image Classification**: Classifies images as "Lionfish" or "Not Lionfish"
- **Flask REST API**: Serves the trained model via HTTP endpoints
- **Interactive Web Viewer**: Browser-based UI for testing the model
- **Real-time Statistics**: Track classification accuracy during testing sessions

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers (32 → 64 → 128 filters)
- MaxPooling layers for downsampling
- Dense layer with 128 units
- Dropout (0.5) for regularization
- Sigmoid output for binary classification

```
Input (256x256x3)
    ↓
Conv2D(32) → MaxPool
    ↓
Conv2D(64) → MaxPool
    ↓
Conv2D(128) → MaxPool
    ↓
Flatten → Dense(128) → Dropout(0.5)
    ↓
Dense(1, sigmoid) → Output
```

## Project Structure

```
lionfish-detection/
├── api/
│   └── lionfish_api.py      # Flask API server
├── model/
│   └── train_model.py       # Model training script
├── web/
│   └── viewer.html          # Web-based testing interface
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/lionfish-detection.git
   cd lionfish-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the trained model**
   
   Download `lionfish_model.h5` from the [Releases](../../releases) page and place it in the `model/` directory.

4. **Set up your dataset** (for training only)
   
   Organize your images in the following structure:
   ```
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
   ```

### Running the API

1. **Update paths** in `api/lionfish_api.py`:
   - Set `MODEL_PATH` to your model location
   - Set `DATASET_PATH` to your dataset location

2. **Start the server**
   ```bash
   python api/lionfish_api.py
   ```

3. **Open the viewer**
   
   Navigate to `http://localhost:5000` in your browser

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface / API info |
| `/health` | GET | Health check status |
| `/classify/random` | GET | Classify a random image from dataset |
| `/classify/random/image` | GET | Get the last classified image |
| `/stats` | GET | Dataset and model statistics |

### Example Response

```json
{
  "predicted_class": "Lionfish",
  "confidence": 0.0234,
  "confidence_percent": "97.66%",
  "true_label": "lionfish",
  "correct": true,
  "split": "test",
  "image_filename": "lionfish_001.jpg"
}
```

## Training Your Own Model

1. Prepare your dataset with the folder structure shown above
2. Update paths in `model/train_model.py`
3. Run the training script:
   ```bash
   python model/train_model.py
   ```

The model uses:
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy
- **Early Stopping**: Patience of 3 epochs
- **Class Weights**: Balanced for handling class imbalance

## Performance

The model achieves strong performance on the test set:
- Precision, Recall, and F1-Score metrics are calculated
- Confusion matrix for detailed analysis
- Training/validation accuracy and loss curves

## Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **Flask** - Web API framework
- **NumPy** - Numerical computing
- **Pillow** - Image processing
- **scikit-learn** - Metrics and evaluation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

