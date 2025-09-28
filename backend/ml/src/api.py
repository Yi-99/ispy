"""
FastAPI-based deployment for car insurance fraud detection
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import io
import base64
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import json
import time
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Model imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models import get_model, UncertaintyEstimator
from .data_preprocessing import DataPreprocessor
from .explainability import FraudDetectionExplainer
from .evaluation import FraudDetectionEvaluator
from config import API_CONFIG, MODELS_DIR, RESULTS_DIR, FRAUD_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Car Insurance Fraud Detection API",
    description="AI-powered fraud detection for car insurance claims",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
loaded_models = {}
model_metadata = {}


class PredictionRequest(BaseModel):
    """Request model for fraud prediction"""
    image_data: str = Field(..., description="Base64 encoded image data")
    model_name: Optional[str] = Field("resnet50", description="Model to use for prediction")
    include_explanation: Optional[bool] = Field(False, description="Include SHAP/LIME explanations")
    threshold: Optional[float] = Field(0.5, description="Fraud detection threshold")


class PredictionResponse(BaseModel):
    """Response model for fraud prediction with probability scoring"""
    fraud_probability: float = Field(..., description="Fraud probability (0.0 to 1.0)")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    confidence: float = Field(..., description="Model confidence (0.0 to 1.0)")
    uncertainty: Optional[float] = Field(None, description="Model uncertainty (0.0 to 1.0)")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="Model used for prediction")
    explanation: Optional[Dict] = Field(None, description="SHAP/LIME explanation if requested")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    image_urls: List[str] = Field(..., description="List of image URLs or base64 data")
    model_name: Optional[str] = Field("resnet50", description="Model to use")
    threshold: Optional[float] = Field(0.5, description="Fraud detection threshold")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[Dict] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total number of images processed")
    processing_time: float = Field(..., description="Total processing time")
    model_used: str = Field(..., description="Model used for predictions")


class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str
    is_loaded: bool
    accuracy: Optional[float] = None
    last_updated: Optional[str] = None
    parameters: Optional[int] = None


def load_model(model_name: str) -> nn.Module:
    """Load a trained model"""
    model_path = MODELS_DIR / f"best_{model_name}_fraud_detector.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
    
    # Load model
    model = get_model(model_name)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Store metadata
    model_metadata[model_name] = {
        'accuracy': checkpoint.get('val_acc', None),
        'last_updated': datetime.now().isoformat(),
        'parameters': sum(p.numel() for p in model.parameters())
    }
    
    logger.info(f"Model {model_name} loaded successfully")
    return model


def preprocess_image(image_data: str) -> torch.Tensor:
    """Preprocess image for model input"""
    try:
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Use torchvision transforms (same as training)
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")


def predict_fraud(model: nn.Module, image_tensor: torch.Tensor, 
                 include_uncertainty: bool = False) -> Dict:
    """Make fraud prediction with probability scoring"""

    model = model.cpu()
    model.eval()
    image_tensor = image_tensor.cpu()
    
    with torch.no_grad():
        # Forward pass
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        # Extract fraud probability (0.0 to 1.0)
        fraud_probability = probabilities[0, 1].item()
        
        # Calculate confidence (how certain the model is)
        confidence = probabilities.max().item()
        
        # Calculate uncertainty if requested
        uncertainty = None
        if include_uncertainty:
            # Calculate entropy as uncertainty measure
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
            uncertainty = entropy[0].item()
        
        # Determine prediction
        prediction = "Fraud" if fraud_probability >= 0.5 else "Non-Fraud"
        
        # Calculate risk level
        risk_level = _calculate_risk_level(fraud_probability)
        
        return {
            "fraud_probability": round(fraud_probability, 4),
            "non_fraud_probability": round(1 - fraud_probability, 4),
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "uncertainty": round(uncertainty, 4) if uncertainty else None,
            "processing_time": 0.0,
            "model_used": "resnet50",
            "explanation": None
        }

def _calculate_risk_level(probability: float) -> str:
    """Convert probability to risk level"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting up fraud detection API...")
    
    # Load default models
    available_models = ['resnet50', 'efficientnet', 'custom_cnn']
    
    for model_name in available_models:
        try:
            model = load_model(model_name)
            loaded_models[model_name] = model
            logger.info(f"Model {model_name} loaded successfully")
        except FileNotFoundError:
            logger.warning(f"Model {model_name} not found, skipping...")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Car Insurance Fraud Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .method { color: #007bff; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Car Insurance Fraud Detection API</h1>
            <p>AI-powered fraud detection for car insurance claims</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> /models - List available models
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> /predict - Single image prediction
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> /predict/batch - Batch predictions
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> /predict/file - Upload file prediction
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> /health - Health check
            </div>
            
            <h2>Documentation:</h2>
            <p><a href="/docs">Interactive API Documentation (Swagger)</a></p>
            <p><a href="/redoc">Alternative Documentation (ReDoc)</a></p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "loaded_models": list(loaded_models.keys()),
        "total_models": len(loaded_models)
    }


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models and their status"""
    models_info = []
    
    for model_name in ['resnet50', 'efficientnet', 'custom_cnn', 'ensemble']:
        is_loaded = model_name in loaded_models
        metadata = model_metadata.get(model_name, {})
        
        models_info.append(ModelInfo(
            model_name=model_name,
            is_loaded=is_loaded,
            accuracy=metadata.get('accuracy'),
            last_updated=metadata.get('last_updated'),
            parameters=metadata.get('parameters')
        ))
    
    return models_info


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud_single(request: PredictionRequest):
    """Predict fraud for a single image"""
    start_time = time.time()
    
    # Validate model
    if request.model_name not in loaded_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model {request.model_name} not available. Available models: {list(loaded_models.keys())}"
        )
    
    try:
        # Preprocess image
        image_tensor = preprocess_image(request.image_data)
        
        # Make prediction
        model = loaded_models[request.model_name]
        model.eval()
        result = predict_fraud(model, image_tensor, include_uncertainty=True)
        
        # Apply threshold
        if result['fraud_probability'] >= request.threshold:
            result['prediction'] = "Fraud"
        else:
            result['prediction'] = "Non-Fraud"
        
        # Add explanation if requested
        explanation = None
        if request.include_explanation:
            try:
                explainer = FraudDetectionExplainer(model)
                # Note: In production, you'd want to cache the explainer setup
                explanation = {"message": "Explanation feature requires model setup"}
            except Exception as e:
                logger.warning(f"Explanation generation failed: {e}")
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            fraud_probability=result['fraud_probability'],
            risk_level=result['risk_level'],
            confidence=result['confidence'],
            uncertainty=result['uncertainty'],
            processing_time=processing_time,
            model_used=request.model_name,
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/file", response_model=PredictionResponse)
async def predict_fraud_file(
    file: UploadFile = File(...),
    model_name: str = "resnet50",
    threshold: float = 0.5,
    include_explanation: bool = False
):
    """Predict fraud for uploaded file"""
    start_time = time.time()
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate model
    if model_name not in loaded_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model {model_name} not available"
        )
    
    try:
        # Read file
        file_content = await file.read()
        
        # Convert to base64
        image_data = base64.b64encode(file_content).decode('utf-8')
        
        # Preprocess image
        image_tensor = preprocess_image(image_data)
        
        # Make prediction
        model = loaded_models[model_name]
        result = predict_fraud(model, image_tensor, include_uncertainty=True)
        
        # Apply threshold
        if result['fraud_probability'] >= threshold:
            result['prediction'] = "Fraud"
        else:
            result['prediction'] = "Non-Fraud"
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            fraud_probability=result['fraud_probability'],
            risk_level=result['risk_level'],
            confidence=result['confidence'],
            uncertainty=result['uncertainty'],
            processing_time=processing_time,
            model_used=model_name,
            explanation=None
        )
        
    except Exception as e:
        logger.error(f"File prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(request: BatchPredictionRequest):
    """Predict fraud for multiple images"""
    start_time = time.time()
    
    # Validate model
    if request.model_name not in loaded_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model {request.model_name} not available"
        )
    
    model = loaded_models[request.model_name]
    predictions = []
    
    for i, image_data in enumerate(request.image_urls):
        try:
            # Preprocess image
            image_tensor = preprocess_image(image_data)
            
            # Make prediction
            result = predict_fraud(model, image_tensor)
            
            # Apply threshold
            if result['fraud_probability'] >= request.threshold:
                result['prediction'] = "Fraud"
            else:
                result['prediction'] = "Non-Fraud"
            
            predictions.append({
                "index": i,
                "prediction": result['prediction'],
                "confidence": result['confidence'],
                "fraud_probability": result['fraud_probability']
            })
            
        except Exception as e:
            logger.error(f"Batch prediction failed for image {i}: {e}")
            predictions.append({
                "index": i,
                "error": str(e)
            })
    
    processing_time = time.time() - start_time
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
        processing_time=processing_time,
        model_used=request.model_name
    )


@app.get("/stats")
async def get_statistics():
    """Get API usage statistics"""
    return {
        "total_models_loaded": len(loaded_models),
        "available_models": list(loaded_models.keys()),
        "api_version": "1.0.0",
        "uptime": "N/A",  # In production, track actual uptime
        "last_updated": datetime.now().isoformat()
    }


@app.post("/models/{model_name}/load")
async def load_model_endpoint(model_name: str):
    """Load a specific model"""
    try:
        if model_name in loaded_models:
            return {"message": f"Model {model_name} already loaded"}
        
        model = load_model(model_name)
        loaded_models[model_name] = model
        
        return {"message": f"Model {model_name} loaded successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a specific model"""
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not loaded")
    
    del loaded_models[model_name]
    if model_name in model_metadata:
        del model_metadata[model_name]
    
    return {"message": f"Model {model_name} unloaded successfully"}


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )


def main():
    """Run the API server"""
    uvicorn.run(
        "api:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["debug"],
        log_level="info"
    )


if __name__ == "__main__":
    main()
