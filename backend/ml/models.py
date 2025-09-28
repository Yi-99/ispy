"""
Central model hub for car insurance fraud detection
Imports all models from separate files to maintain modularity
"""
import torch
import torch.nn as nn
import logging

# Import base components
from models.base_models import (
    FocalLoss, 
    UncertaintyEstimator, 
    count_parameters, 
    get_available_models
)

# Import individual model architectures
from models.resnet_model import get_resnet_model
from models.efficientnet_model import get_efficientnet_model
from models.custom_cnn_model import get_custom_cnn_model
from models.vit_model import get_vit_model
from models.ensemble_model import get_ensemble_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model(model_name: str, num_classes: int = 2, **kwargs) -> nn.Module:
    """Factory function to get different models"""
    
    models_dict = {
        'custom_cnn': get_custom_cnn_model,
        'resnet50': get_resnet_model,
        'efficientnet': get_efficientnet_model,
        'vit': get_vit_model,
        'ensemble': get_ensemble_model
    }
    
    if model_name not in models_dict:
        available_models = list(models_dict.keys())
        raise ValueError(f"Model {model_name} not supported. Available models: {available_models}")
    
    return models_dict[model_name](num_classes=num_classes, **kwargs)


def main():
    """Test all models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different models
    model_names = get_available_models()
    
    for model_name in model_names:
        print(f"\nTesting {model_name}:")
        try:
            model = get_model(model_name)
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