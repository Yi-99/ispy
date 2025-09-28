# debug_api_flow.py
import torch
import base64
from models import get_model
from src.api import preprocess_image

print("=== Step 1: Load Model (Same as API) ===")
model = get_model('resnet50', num_classes=2)
checkpoint = torch.load('results/best_resnet50_fraud_detector.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✅ Model loaded and set to eval mode")

print("\n=== Step 2: Test with Dummy Input (Same as debug_model.py) ===")
dummy_input = torch.randn(1, 3, 224, 224)
try:
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✅ Dummy input works! Output shape: {output.shape}")
except Exception as e:
    print(f"❌ Dummy input failed: {e}")

print("\n=== Step 3: Test with API Preprocessing ===")
# Convert image to base64 (same as test_prediction.py)
with open('predictions/test_images/63.jpg', 'rb') as f:
    encoded = base64.b64encode(f.read()).decode('utf-8')

try:
    # Use the same preprocessing as API
    image_tensor = preprocess_image(encoded)
    print(f"✅ Preprocessing works! Shape: {image_tensor.shape}, Type: {image_tensor.dtype}")
    
    # Test with preprocessed image
    with torch.no_grad():
        output = model(image_tensor)
    print(f"✅ API preprocessing works! Output shape: {output.shape}")
    
except Exception as e:
    print(f"❌ API preprocessing failed: {e}")
    print(f"Error type: {type(e)}")
    print(f"Error details: {str(e)}")

# Add this at the end of debug_api_flow.py
print("\n=== Step 4: Check Device Information ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")

# Check if model is on GPU
if next(model.parameters()).is_cuda:
    print("⚠️  Model is on GPU - this might be the issue!")
    print("Moving model to CPU...")
    model = model.cpu()
    model.eval()
    
    # Test again
    try:
        with torch.no_grad():
            output = model(image_tensor)
        print(f"✅ CPU model works! Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ CPU model failed: {e}")
else:
    print("✅ Model is on CPU")