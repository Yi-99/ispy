"""
Test the trained fraud detection model on the test dataset
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models import get_model
from src.data_preprocessing import DataPreprocessor
from config import DATASET_CONFIG

def test_model_accuracy():
    """Test the trained model on the test dataset"""
    
    print("🚗 Car Insurance Fraud Detection - Model Testing")
    print("=" * 60)
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor()
    
    # Create test data loader
    print("📁 Loading test dataset...")
    _, test_loader = preprocessor.create_data_loaders(
        DATASET_CONFIG["train_dir"],  # We only need test data
        DATASET_CONFIG["test_dir"]
    )
    
    # Load the trained model
    print("🤖 Loading trained ResNet50 model...")
    model_path = "models/best_resnet50_fraud_detector.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        return
    
    try:
        # Create model
        model = get_model('resnet50', num_classes=2)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"   Training Info:")
        print(f"   - Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
        print(f"   - Best Epoch: {checkpoint.get('epoch', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test the model
    print("\n🧪 Testing model on test dataset...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    correct_predictions = 0
    total_predictions = 0
    fraud_correct = 0
    fraud_total = 0
    non_fraud_correct = 0
    non_fraud_total = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            # Get predictions
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            
            # Calculate metrics
            batch_correct = (predictions == target).sum().item()
            correct_predictions += batch_correct
            total_predictions += target.size(0)
            
            # Detailed metrics
            for i in range(target.size(0)):
                true_label = target[i].item()
                pred_label = predictions[i].item()
                
                if true_label == 1:  # Fraud case
                    fraud_total += 1
                    if pred_label == 1:
                        fraud_correct += 1
                    else:
                        false_negatives += 1
                else:  # Non-fraud case
                    non_fraud_total += 1
                    if pred_label == 0:
                        non_fraud_correct += 1
                    else:
                        false_positives += 1
    
    # Calculate final metrics
    accuracy = correct_predictions / total_predictions
    fraud_detection_rate = fraud_correct / fraud_total if fraud_total > 0 else 0
    false_positive_rate = false_positives / non_fraud_total if non_fraud_total > 0 else 0
    false_negative_rate = false_negatives / fraud_total if fraud_total > 0 else 0
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    print(f"📈 Overall Performance:")
    print(f"   - Total Test Samples: {total_predictions:,}")
    print(f"   - Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\n🎯 Fraud Detection Performance:")
    print(f"   - Fraud Cases Detected: {fraud_correct}/{fraud_total} ({fraud_detection_rate:.4f})")
    print(f"   - Fraud Detection Rate: {fraud_detection_rate*100:.2f}%")
    print(f"   - False Negatives (Missed Fraud): {false_negatives}/{fraud_total} ({false_negative_rate*100:.2f}%)")
    
    print(f"\n✅ Legitimate Claims Performance:")
    print(f"   - Non-Fraud Cases Correct: {non_fraud_correct}/{non_fraud_total} ({non_fraud_correct/non_fraud_total:.4f})")
    print(f"   - False Positives (False Alarms): {false_positives}/{non_fraud_total} ({false_positive_rate*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("✅ Testing completed!")

if __name__ == "__main__":
    test_model_accuracy()