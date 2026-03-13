# Banknote Authentication with Docker

A containerized machine learning application that detects counterfeit banknotes using image-extracted features.

## Changes from Original Lab

- **Dataset**: Changed from Iris (sklearn) to Banknote Authentication (UCI)
- **Task**: Changed from flower classification (3 classes) to counterfeit detection (binary)
- **Framework**: Changed from TensorFlow to PyTorch
- **UI**: Changed from form with dropdowns to sliders with real-time values
- **Output**: Changed from species name to Authentic/Fake with confidence score

## Project Structure

```
Docker_Lab/
├── src/
│   ├── model_training.py    # PyTorch model training
│   ├── main.py              # Flask API server
│   ├── templates/
│   │   └── predict.html     # Web interface
│   └── statics/
├── docker-compose.yml       # Multi-container setup
├── dockerfile
├── requirements.txt
└── README.md
```

## How It Works

1. **Training Container**: Fetches UCI Banknote dataset, trains a PyTorch neural network, saves model and scaler to shared volume
2. **Serving Container**: Loads trained model, runs Flask API on port 4000

## Model Architecture

- Input: 4 features (variance, skewness, kurtosis, entropy)
- Hidden: 16 → 8 neurons with ReLU
- Output: Sigmoid (binary classification)
- Optimizer: Adam (lr=0.01)
- Loss: Binary Cross-Entropy

## Run

```bash
docker compose up --build
```

Visit: http://localhost:4000

## Dataset

UCI Banknote Authentication Dataset (ID: 267)
- 1372 samples
- Features extracted from banknote images using Wavelet Transform
- Binary target: 0 = Authentic, 1 = Fake