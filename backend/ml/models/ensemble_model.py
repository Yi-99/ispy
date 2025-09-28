"""
Ensemble model combining multiple fraud detection models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_models import count_parameters
from .resnet_model import get_resnet_model
from .efficientnet_model import get_efficientnet_model
from .custom_cnn_model import get_custom_cnn_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleFraudDetector(nn.Module):
    """Ensemble model combining multiple architectures"""
    
    def __init__(self, num_classes: int = 2, model_names: list = None, dropout_rate: float = 0.5):
        super(EnsembleFraudDetector, self).__init__()
        
        if model_names is None:
            model_names = ['resnet50', 'efficientnet', 'custom_cnn']
        
        self.models = nn.ModuleDict()
        
        # Load individual models
        for model_name in model_names:
            if model_name == 'resnet50':
                self.models[model_name] = get_resnet_model(num_classes=num_classes)
            elif model_name == 'efficientnet':
                self.models[model_name] = get_efficientnet_model(num_classes=num_classes)
            elif model_name == 'custom_cnn':
                self.models[model_name] = get_custom_cnn_model(num_classes=num_classes)
        
        # Ensemble classifier
        num_models = len(model_names)
        self.ensemble_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_models * num_classes, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Get predictions from all models
        model_outputs = []
        for model_name, model in self.models.items():
            output = model(x)
            model_outputs.append(output)
        
        # Concatenate outputs
        ensemble_input = torch.cat(model_outputs, dim=1)
        
        # Final prediction
        final_output = self.ensemble_classifier(ensemble_input)
        
        return final_output


def get_ensemble_model(num_classes: int = 2, **kwargs):
    """Factory function to get Ensemble model"""
    model = EnsembleFraudDetector(num_classes=num_classes, **kwargs)
    logger.info(f"Created Ensemble model with {count_parameters(model):,} parameters")
    return model


def main():
    """Test the Ensemble model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Ensemble Fraud Detector:")
    try:
        model = get_ensemble_model()
        model = model.to(device)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        output = model(dummy_input)
        
        print(f"  Parameters: {count_parameters(model):,}")
        print(f"  Output shape: {output.shape}")
        print(f"  ✓ Model working correctly")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    main()
