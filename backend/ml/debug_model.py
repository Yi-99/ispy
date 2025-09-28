# debug_model.py
import torch
import torch.nn as nn
from models import get_model

# Load model
model = get_model('resnet50', num_classes=2)
checkpoint = torch.load('results/best_resnet50_fraud_detector.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Set to eval mode
model.eval()

# Test with dummy input
dummy_input = torch.randn(1, 3, 224, 224)
print(f"Input shape: {dummy_input.shape}")

try:
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✅ Success! Output shape: {output.shape}")
    print(f"Output: {output}")
except Exception as e:
    print(f"❌ Error: {e}")
    
    # Try to identify which layer is causing the issue
    print("\nDebugging model layers:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)):
            print(f"  {name}: {type(module).__name__} - training={module.training}")