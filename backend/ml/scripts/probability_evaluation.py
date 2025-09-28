"""
Probability-based fraud detection evaluation
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models import get_model
from src.data_preprocessing import DataPreprocessor
from config import DATASET_CONFIG

def probability_evaluation():
    """Evaluate model using probability-based scoring"""
    
    print("🎯 PROBABILITY-BASED FRAUD DETECTION EVALUATION")
    print("=" * 60)
    
    # Load model and data
    model_path = "models/best_resnet50_fraud_detector.pth"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = get_model('resnet50', num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    preprocessor = DataPreprocessor()
    _, test_loader = preprocessor.create_data_loaders(
        DATASET_CONFIG["train_dir"],
        DATASET_CONFIG["test_dir"]
    )
    
    # Get all predictions
    all_probabilities = []
    all_targets = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    print(f"📊 PROBABILITY DISTRIBUTION:")
    fraud_probs = all_probabilities[all_targets == 1]
    non_fraud_probs = all_probabilities[all_targets == 0]
    
    print(f"   Fraud Cases:")
    print(f"     - Mean Probability: {np.mean(fraud_probs):.4f}")
    print(f"     - Median: {np.median(fraud_probs):.4f}")
    print(f"     - Range: {np.min(fraud_probs):.4f} - {np.max(fraud_probs):.4f}")
    
    print(f"   Non-Fraud Cases:")
    print(f"     - Mean Probability: {np.mean(non_fraud_probs):.4f}")
    print(f"     - Median: {np.median(non_fraud_probs):.4f}")
    print(f"     - Range: {np.min(non_fraud_probs):.4f} - {np.max(non_fraud_probs):.4f}")
    
    # Risk level analysis
    print(f"\n🎚️  RISK LEVEL ANALYSIS:")
    
    risk_levels = [
        (0.0, 0.3, "Low Risk"),
        (0.3, 0.7, "Medium Risk"),
        (0.7, 1.0, "High Risk")
    ]
    
    for low, high, level in risk_levels:
        mask = (all_probabilities >= low) & (all_probabilities < high)
        total_in_level = np.sum(mask)
        fraud_in_level = np.sum(mask & (all_targets == 1))
        non_fraud_in_level = np.sum(mask & (all_targets == 0))
        
        fraud_rate = fraud_in_level / total_in_level if total_in_level > 0 else 0
        
        print(f"   {level} ({low:.1f}-{high:.1f}):")
        print(f"     - Total Cases: {total_in_level}")
        print(f"     - Fraud Cases: {fraud_in_level}")
        print(f"     - Non-Fraud Cases: {non_fraud_in_level}")
        print(f"     - Actual Fraud Rate: {fraud_rate:.3f} ({fraud_rate*100:.1f}%)")
    
    # Show examples
    print(f"\n📋 PROBABILITY EXAMPLES:")
    
    # High probability fraud cases
    high_prob_fraud = np.where((all_probabilities > 0.7) & (all_targets == 1))[0][:5]
    print(f"   🔴 High Probability Fraud Cases (Actual Fraud):")
    for i, idx in enumerate(high_prob_fraud):
        prob = all_probabilities[idx]
        print(f"      Case {i+1}: Probability = {prob:.3f} (High Risk)")
    
    # Low probability fraud cases (missed)
    low_prob_fraud = np.where((all_probabilities < 0.3) & (all_targets == 1))[0][:5]
    print(f"   🟡 Low Probability Fraud Cases (Missed Fraud):")
    for i, idx in enumerate(low_prob_fraud):
        prob = all_probabilities[idx]
        print(f"      Case {i+1}: Probability = {prob:.3f} (Low Risk - MISSED!)")
    
    # High probability non-fraud (false alarms)
    high_prob_non_fraud = np.where((all_probabilities > 0.7) & (all_targets == 0))[0][:5]
    print(f"   🟠 High Probability Non-Fraud Cases (False Alarms):")
    for i, idx in enumerate(high_prob_non_fraud):
        prob = all_probabilities[idx]
        print(f"      Case {i+1}: Probability = {prob:.3f} (High Risk - FALSE ALARM!)")
    
    return all_probabilities, all_targets

if __name__ == "__main__":
    probabilities, targets = probability_evaluation()