"""
ResNet-based fraud detection model
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from .base_models import count_parameters
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResNetFraudDetector(nn.Module):
    """ResNet-based fraud detector with custom classifier"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.5):
        super(ResNetFraudDetector, self).__init__()
        
        # Load pretrained ResNet50
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Remove the original classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Custom classifier for fraud detection
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Extract features using ResNet backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Classify
        output = self.classifier(features)
        return output


def get_resnet_model(num_classes: int = 2, **kwargs):
    """Factory function to get ResNet model"""
    model = ResNetFraudDetector(num_classes=num_classes, **kwargs)
    logger.info(f"Created ResNet model with {count_parameters(model):,} parameters")
    return model


def main():
    """Test the ResNet model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing ResNet50 Fraud Detector:")
    try:
        model = get_resnet_model()
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
