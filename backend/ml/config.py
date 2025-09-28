"""
Configuration file for Car Insurance Fraud Detection System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "train_dir": DATA_DIR / "train",
    "test_dir": DATA_DIR / "test",
    "image_size": (224, 224),
    "batch_size": 32,
    "num_workers": 4,
    "class_names": ["Non-Fraud", "Fraud"],
    "fraud_class": 1,
    "non_fraud_class": 0
}

# Model configuration
MODEL_CONFIG = {
    "input_size": (224, 224, 3),
    "num_classes": 2,
    "learning_rate": 0.001,
    "epochs": 50,
    "patience": 10,
    "min_delta": 0.001
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    "resize": (224, 224),
    "normalize": True,
    "augmentation": {
        "rotation_range": 20,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "horizontal_flip": True,
        "zoom_range": 0.2,
        "brightness_range": [0.8, 1.2],
        "contrast_range": [0.8, 1.2]
    }
}

# Fraud detection specific configuration
FRAUD_CONFIG = {
    "threshold": 0.5,
    "uncertainty_threshold": 0.3,
    "ensemble_weights": {
        "cnn": 0.4,
        "traditional_ml": 0.3,
        "feature_engineering": 0.3
    },
    "explainability": {
        "use_shap": True,
        "use_lime": True,
        "top_features": 10
    }
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp"]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "fraud_detection.log"
}
