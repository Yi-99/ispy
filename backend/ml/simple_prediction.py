# simple_prediction.py
import torch
import base64
from models import get_model
from src.api import preprocess_image

# Load model exactly like the API does
model = get_model('resnet50', num_classes=2)
checkpoint = torch.load('results/best_resnet50_fraud_detector.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Convert image to base64 (same as API)
with open('predictions/test_images/63.jpg', 'rb') as f:
    encoded = base64.b64encode(f.read()).decode('utf-8')

# Preprocess image (same as API)
image_tensor = preprocess_image(encoded)

# Make prediction (same as API predict_fraud function)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
image_tensor = image_tensor.to(device)

with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    fraud_probability = probabilities[0, 1].item()
    confidence = probabilities.max().item()

print(f"✅ SUCCESS!")
print(f"Fraud Probability: {fraud_probability:.4f}")
print(f"Non-Fraud Probability: {1-fraud_probability:.4f}")
print(f"Confidence: {confidence:.4f}")
print(f"Prediction: {'Fraud' if fraud_probability > 0.5 else 'Non-Fraud'}")
