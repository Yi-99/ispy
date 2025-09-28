"""
Vision Transformer model for fraud detection
"""
import torch
import torch.nn as nn
import timm
from .base_models import count_parameters
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionTransformerFraudDetector(nn.Module):
    """Vision Transformer for fraud detection"""
    
    def __init__(self, num_classes: int = 2, model_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True, dropout_rate: float = 0.5):
        super(VisionTransformerFraudDetector, self).__init__()
        
        # Load pretrained ViT
        if pretrained:
            self.backbone = timm.create_model(model_name, pretrained=True)
        else:
            self.backbone = timm.create_model(model_name, pretrained=False)
        
        # Get the number of features from the backbone
        num_features = self.backbone.head.in_features
        
        # Replace head
        self.backbone.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
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
        return self.backbone(x)


def get_vit_model(num_classes: int = 2, **kwargs):
    """Factory function to get Vision Transformer model"""
    model = VisionTransformerFraudDetector(num_classes=num_classes, **kwargs)
    logger.info(f"Created Vision Transformer model with {count_parameters(model):,} parameters")
    return model


def main():
    """Test the Vision Transformer model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Vision Transformer Fraud Detector:")
    try:
        model = get_vit_model()
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
