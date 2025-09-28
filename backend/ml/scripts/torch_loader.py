import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models import get_model

def load_and_test_model():
    """Load the trained ResNet50 fraud detection model and test it"""
    
    print("Loading trained ResNet50 fraud detection model...")
    
    try:
        # Create the model architecture
        model = get_model('resnet50', num_classes=2)
        
        # Load the trained weights
        checkpoint_path = 'best_resnet50_fraud_detector.pth'
        if not os.path.exists(checkpoint_path):
            print(f"Error: Model file not found at {checkpoint_path}")
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("✅ Model loaded successfully!")
        print(f"Model info:")
        print(f"  - Validation Accuracy: {checkpoint.get('val_acc', 'N/A')}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Model Name: {checkpoint.get('model_name', 'resnet50')}")
        
        # Test the model with a dummy input
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1)
            
        print(f"✅ Model test successful!")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Probabilities: {probabilities[0].tolist()}")
        print(f"  - Prediction: {'Fraud' if prediction.item() == 1 else 'Non-Fraud'}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

if __name__ == "__main__":
    model = load_and_test_model()
    if model:
        print("\n🎉 Model is ready to use!")
    else:
        print("\n💥 Failed to load model")