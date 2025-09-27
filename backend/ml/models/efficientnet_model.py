"""
EfficientNet-based fraud detection model
"""
import torch
import torch.nn as nn
import timm
from .base_models import count_parameters
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EfficientNetFraudDetector(nn.Module):
    """EfficientNet-based fraud detector"""
    
    def __init__(self, num_classes: int = 2, model_name: str = 'efficientnet_b0', 
                 pretrained: bool = True, dropout_rate: float = 0.5):
        super(EfficientNetFraudDetector, self).__init__()
        
        # Load pretrained EfficientNet
        if pretrained:
            self.backbone = timm.create_model(model_name, pretrained=True)
        else:
            self.backbone = timm.create_model(model_name, pretrained=False)
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier.in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


def get_efficientnet_model(num_classes: int = 2, **kwargs):
    """Factory function to get EfficientNet model"""
    model = EfficientNetFraudDetector(num_classes=num_classes, **kwargs)
    logger.info(f"Created EfficientNet model with {count_parameters(model):,} parameters")
    return model


def main():
    """Test the EfficientNet model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing EfficientNet Fraud Detector:")
    try:
        model = get_efficientnet_model()
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
