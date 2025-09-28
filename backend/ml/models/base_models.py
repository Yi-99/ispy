"""
Base models and shared components for car insurance fraud detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on important regions"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Channel-wise average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_combined = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid
        x_attention = self.conv(x_combined)
        x_attention = self.sigmoid(x_attention)
        
        return x * x_attention


class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Global average pooling
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out)
        
        # Global max pooling
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out)
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        
        return x * out


class UncertaintyEstimator(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation
    """
    def __init__(self, base_model, num_samples=10):
        super(UncertaintyEstimator, self).__init__()
        self.base_model = base_model
        self.num_samples = num_samples
    
    def forward(self, x):
        # Enable dropout during inference
        self.base_model.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = self.base_model(x)
                predictions.append(F.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        
        # Calculate epistemic uncertainty (entropy)
        uncertainty = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=1)
        
        return mean_pred, uncertainty, var_pred


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_available_models() -> List[str]:
    """Get list of available model names"""
    return ['custom_cnn', 'resnet50', 'efficientnet', 'vit', 'ensemble']
