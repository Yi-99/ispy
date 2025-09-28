"""
Models package for car insurance fraud detection
"""
import torch
import torch.nn as nn

from .base_models import (
    FocalLoss, 
    UncertaintyEstimator, 
    count_parameters, 
    get_available_models
)
from .resnet_model import get_resnet_model
from .efficientnet_model import get_efficientnet_model
from .custom_cnn_model import get_custom_cnn_model
from .vit_model import get_vit_model
from .ensemble_model import get_ensemble_model


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


__all__ = [
    'FocalLoss',
    'UncertaintyEstimator', 
    'count_parameters',
    'get_available_models',
    'get_model',
    'get_resnet_model',
    'get_efficientnet_model',
    'get_custom_cnn_model',
    'get_vit_model',
    'get_ensemble_model'
]
