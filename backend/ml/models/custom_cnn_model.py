"""
Custom CNN model with attention mechanism for fraud detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_models import SpatialAttention, count_parameters
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionCNN(nn.Module):
    """Custom CNN for fraud detection with attention mechanism"""
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super(FraudDetectionCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Attention mechanism
        self.attention = SpatialAttention()
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        # Apply attention
        x = self.attention(x)
        
        # Global pooling and classification
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


def get_custom_cnn_model(num_classes: int = 2, **kwargs):
    """Factory function to get Custom CNN model"""
    model = FraudDetectionCNN(num_classes=num_classes, **kwargs)
    logger.info(f"Created Custom CNN model with {count_parameters(model):,} parameters")
    return model


def main():
    """Test the Custom CNN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Custom CNN Fraud Detector:")
    try:
        model = get_custom_cnn_model()
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
