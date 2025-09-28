# 🚗 Car Insurance Fraud Detection System

A comprehensive deep learning system for detecting fraudulent car insurance claims using computer vision and advanced machine learning techniques.

## 🎯 Overview

This system uses state-of-the-art deep learning models to analyze car damage images and detect potential fraud cases. It combines multiple approaches including ResNet50, EfficientNet, Vision Transformers, and ensemble methods with fraud-specific optimization techniques.

## ✨ Key Features

- **🔍 Multi-Model Architecture**: ResNet50, EfficientNet, Custom CNN, Vision Transformer, and Ensemble models
- **⚖️ Class Imbalance Handling**: Focal Loss, Class-Balanced Loss, and weighted sampling
- **🎨 Advanced Augmentation**: 8x fraud-specific augmentation with weather effects and distortions
- **📊 Comprehensive Evaluation**: Fraud-focused metrics, threshold analysis, and business cost optimization
- **🔬 Explainable AI**: SHAP and LIME integration for model interpretability
- **🚀 Production Ready**: FastAPI endpoints with uncertainty estimation
- **📈 Hybrid Approach**: CNN feature extraction + traditional ML classifiers

## 📁 Project Structure

```
fraud_detection/
├── main.py                    # 🎯 Entry point with --enhanced flag
├── models.py                  # 🎯 Model hub importing from models/ package
├── config.py                  # 🎯 Configuration settings
├── requirements.txt           # 🎯 Python dependencies
│
├── models/                    # 🤖 Model package
│   ├── base_models.py         # Shared components (FocalLoss, Attention, etc.)
│   ├── resnet_model.py        # ResNet50-based fraud detector
│   ├── efficientnet_model.py  # EfficientNet-based fraud detector
│   ├── custom_cnn_model.py    # Custom CNN with attention mechanism
│   ├── vit_model.py           # Vision Transformer model
│   └── ensemble_model.py      # Ensemble of multiple models
│
├── src/                       # 🔧 Core source code
│   ├── training.py            # Enhanced training pipeline
│   ├── data_preprocessing.py  # Enhanced data preprocessing
│   ├── evaluation.py          # Comprehensive evaluation framework
│   ├── explainability.py      # SHAP/LIME explainability
│   ├── hybrid_approach.py     # Hybrid CNN+ML approach
│   └── api.py                 # FastAPI endpoints
│
├── scripts/                   # 📜 Testing and utility scripts
│   ├── test_accuracy.py       # Model testing script
│   ├── probability_evaluation.py # Probability-based evaluation
│   └── torch_loader.py        # Model loading utility
│
├── legacy/                    # 📦 Deprecated/superseded files
├── docs/                      # 📚 Documentation
├── data/                      # 📊 Dataset
│   ├── train/
│   │   ├── Fraud/             # Fraud cases (200 images)
│   │   └── Non-Fraud/         # Legitimate cases (5,000 images)
│   └── test/
│       ├── Fraud/             # Test fraud cases (93 images)
│       └── Non-Fraud/         # Test legitimate cases (1,323 images)
│
├── results/                   # 📈 Training results and saved models
└── logs/                      # 📝 Training logs
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd fraud_detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Enhanced Training

```bash
# Train with enhanced fraud detection pipeline
python3 main.py train --enhanced

# Train specific model
python3 main.py train --enhanced --model resnet50
python3 main.py train --enhanced --model efficientnet
python3 main.py train --enhanced --model custom_cnn
```

### 3. Model Evaluation

```bash
# Comprehensive evaluation
python3 main.py evaluate

# Test model accuracy
python3 scripts/test_accuracy.py

# Probability-based evaluation
python3 scripts/probability_evaluation.py
```

### 4. API Server

```bash
# Start API server
python3 main.py api --port 8000

# Access API documentation at http://localhost:8000/docs
```

## 🎯 Enhanced Training Features

### **Fraud-Specific Optimization**
- **Class-Balanced Loss**: Handles severe class imbalance (3.8% fraud rate)
- **8x Fraud Augmentation**: Aggressive augmentation specifically for fraud cases
- **Weighted Sampling**: Ensures fraud cases appear in every batch
- **Fraud-Focused Metrics**: Prioritizes fraud detection over overall accuracy

### **Advanced Augmentation**
- **Weather Effects**: Fog, rain, sun flare simulation
- **Distortion Effects**: Elastic, grid, optical distortion
- **Quality Degradation**: Compression, downscaling, noise
- **Environmental Effects**: Lighting variations, shadows

### **Enhanced Early Stopping**
- **Higher Patience**: 25 epochs vs 5 for better fraud detection
- **Composite Scoring**: Combines fraud detection rate, F1-score, probability separation, and AUC
- **Business Cost Optimization**: Minimizes total business cost (missed fraud + false alarms)

## 📊 Expected Performance

With the enhanced training pipeline, you should see:

- **Fraud Detection Rate**: 70-85% (vs baseline 64.5%)
- **Probability Separation**: 10-20% (vs baseline 1.94%)
- **Training Duration**: 20-50 epochs (vs baseline 4)
- **High Confidence Predictions**: More confident fraud identification

## 🛠️ Available Models

| Model | File | Parameters | Best For |
|-------|------|------------|----------|
| **ResNet50** | `models/resnet_model.py` | ~25M | Production use |
| **EfficientNet** | `models/efficientnet_model.py` | ~5M | Mobile/Edge |
| **Custom CNN** | `models/custom_cnn_model.py` | ~1M | Quick experiments |
| **Vision Transformer** | `models/vit_model.py` | ~86M | Research |
| **Ensemble** | `models/ensemble_model.py` | ~31M | Best accuracy |

## 📈 Usage Examples

### **Training**
```python
from src import EnhancedFraudTrainer, EnhancedDataPreprocessor
from models import get_model

# Initialize enhanced preprocessor
preprocessor = EnhancedDataPreprocessor(
    fraud_augmentation_factor=8,
    use_enhanced_augmentation=True
)

# Create enhanced data loaders
train_loader, test_loader = preprocessor.create_data_loaders(
    train_dir, test_dir, use_enhanced=True
)

# Train with enhanced pipeline
trainer = EnhancedFraudTrainer(model_name='resnet50')
trainer.train(train_loader, test_loader)
```

### **Evaluation**
```python
from src import FraudDetectionEvaluator

# Initialize evaluator
evaluator = FraudDetectionEvaluator()

# Comprehensive evaluation
test_results, probabilities, targets = trainer.evaluate_with_thresholds(test_loader)
optimal_threshold = trainer.plot_comprehensive_analysis(test_results, probabilities, targets)
```

### **API Usage**
```python
import requests

# Upload image for fraud detection
with open('damage_image.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/predict', files={'file': f})

result = response.json()
print(f"Fraud Probability: {result['fraud_probability']:.3f}")
print(f"Risk Level: {result['risk_level']}")
```

## 🔧 Configuration

### **Model Configuration** (`config.py`)
```python
MODEL_CONFIG = {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "patience": 25,  # Enhanced for fraud detection
}

DATASET_CONFIG = {
    "train_dir": "data/train",
    "test_dir": "data/test",
    "class_names": ["Non-Fraud", "Fraud"],
    "fraud_class": 1,
    "non_fraud_class": 0,
}
```

## 📊 Evaluation Metrics

### **Fraud-Specific Metrics**
- **Fraud Detection Rate**: Percentage of fraud cases correctly identified
- **False Positive Rate**: Percentage of legitimate cases incorrectly flagged
- **Probability Separation**: Difference between fraud and non-fraud probabilities
- **Business Cost**: Total cost considering missed fraud and false alarms

### **Comprehensive Analysis**
- **Threshold Analysis**: Performance across different probability thresholds
- **ROC Curves**: True positive vs false positive rates
- **Precision-Recall Curves**: Precision vs recall for fraud detection
- **Business Impact**: ROI and cost optimization analysis

## 🚀 Production Deployment

### **FastAPI Server**
```bash
# Start production server
python3 main.py api --port 8000

# With gunicorn for production
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api:app --bind 0.0.0.0:8000
```

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python3", "main.py", "api", "--port", "8000"]
```

## 🧪 Testing

### **Run All Tests**
```bash
# Test model accuracy
python3 scripts/test_accuracy.py

# Probability evaluation
python3 scripts/probability_evaluation.py

# Load and test saved model
python3 scripts/torch_loader.py
```

### **Custom Testing**
```python
from models import get_model
from src.data_preprocessing import DataPreprocessor

# Load trained model
model = get_model('resnet50')
checkpoint = torch.load('results/best_resnet50_fraud_detector.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Test on new images
preprocessor = DataPreprocessor()
# ... test your images
```

## 📚 Documentation

- **Enhanced Training Guide**: `docs/ENHANCED_TRAINING_GUIDE.md`
- **Data Preprocessing Guide**: `docs/ENHANCED_DATA_PREPROCESSING_GUIDE.md`
- **Modular Models Guide**: `docs/MODULAR_MODELS_GUIDE.md`
- **Project Organization**: `docs/PROJECT_ORGANIZATION.md`

## 🔍 Troubleshooting

### **Common Issues**

1. **Import Errors**: Make sure you're running from the project root directory
2. **CUDA Out of Memory**: Reduce batch size in `config.py`
3. **Poor Performance**: Ensure you're using the enhanced training pipeline with `--enhanced` flag

### **Performance Issues**

1. **Slow Training**: Reduce `fraud_augmentation_factor` from 8 to 5
2. **Memory Issues**: Disable enhanced augmentation with `use_enhanced_augmentation=False`
3. **Early Stopping**: Increase patience in `MODEL_CONFIG`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: Car Parts and Car Damages dataset
- **Models**: PyTorch, torchvision, timm
- **Augmentation**: albumentations, imgaug
- **Explainability**: SHAP, LIME
- **API**: FastAPI, uvicorn

## 📞 Support

For questions or issues:
1. Check the documentation in `docs/`
2. Review the troubleshooting section
3. Open an issue on GitHub
4. Contact the development team

---

**🎉 Ready to detect fraud with state-of-the-art AI!**
