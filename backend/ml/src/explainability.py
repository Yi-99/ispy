"""
Explainability module for fraud detection using SHAP and LIME
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging

# SHAP imports
import shap
from shap import Explainer, GradientExplainer, DeepExplainer

# LIME imports
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

# Other imports
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import RESULTS_DIR, FRAUD_CONFIG
from models import get_model
from .data_preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionExplainer:
    """Comprehensive explainability for fraud detection models"""
    
    def __init__(self, model: nn.Module, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.gradient_explainer = None
        
        logger.info("Fraud detection explainer initialized")
    
    def setup_shap_explainer(self, background_data: torch.Tensor, method: str = 'gradient'):
        """Setup SHAP explainer for the model"""
        
        if method == 'gradient':
            self.shap_explainer = GradientExplainer(self.model, background_data)
        elif method == 'deep':
            self.shap_explainer = DeepExplainer(self.model, background_data)
        else:
            raise ValueError(f"Unsupported SHAP method: {method}")
        
        logger.info(f"SHAP explainer setup with {method} method")
    
    def setup_lime_explainer(self, segmentation_fn=None):
        """Setup LIME explainer for image analysis"""
        
        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)
        
        self.lime_explainer = lime_image.LimeImageExplainer(segmentation_fn=segmentation_fn)
        logger.info("LIME explainer setup complete")
    
    def explain_with_shap(self, input_data: torch.Tensor, 
                         class_idx: int = 1, 
                         num_samples: int = 50) -> Dict:
        """Generate SHAP explanations for fraud detection"""
        
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not setup. Call setup_shap_explainer first.")
        
        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(
            input_data, 
            nsamples=num_samples
        )
        
        # Handle different model outputs
        if isinstance(shap_values, list):
            # Multi-class model
            shap_values = shap_values[class_idx]
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model(input_data)
            probabilities = torch.softmax(prediction, dim=1)
            predicted_class = prediction.argmax(dim=1)
            confidence = probabilities.max(dim=1)[0]
        
        explanation = {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'prediction': predicted_class.cpu().detach().numpy(),
            'probabilities': probabilities.cpu().detach().numpy(),
            'confidence': confidence.cpu().detach().numpy(),
            'input_data': input_data.cpu().detach().numpy()
        }
        
        return explanation
    
    def explain_with_lime(self, input_data: torch.Tensor, 
                         num_features: int = 10,
                         num_samples: int = 1000) -> Dict:
        """Generate LIME explanations for fraud detection"""
        
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not setup. Call setup_lime_explainer first.")
        
        # Convert tensor to numpy array
        if isinstance(input_data, torch.Tensor):
            input_array = input_data.cpu().detach().numpy()
        else:
            input_array = input_data
        
        # Handle batch input
        if len(input_array.shape) == 4:
            input_array = input_array[0]  # Take first image
        
        # Transpose from CHW to HWC for LIME
        if input_array.shape[0] == 3:
            input_array = np.transpose(input_array, (1, 2, 0))
        
        # Denormalize image for LIME
        input_array = self._denormalize_image(input_array)
        
        # Define prediction function
        def predict_fn(images):
            # Convert back to tensor format
            batch = []
            for img in images:
                # Normalize and transpose
                img_normalized = self._normalize_image(img)
                img_tensor = torch.from_numpy(img_normalized).unsqueeze(0)
                batch.append(img_tensor)
            
            batch_tensor = torch.cat(batch, dim=0).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(batch_tensor)
                probabilities = torch.softmax(predictions, dim=1)
                return probabilities.cpu().detach().numpy()
        
        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            input_array,
            predict_fn,
            top_labels=2,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Get prediction
        with torch.no_grad():
            input_tensor = torch.from_numpy(self._normalize_image(input_array)).unsqueeze(0).to(self.device)
            prediction = self.model(input_tensor)
            probabilities = torch.softmax(prediction, dim=1)
            predicted_class = prediction.argmax(dim=1)
            confidence = probabilities.max(dim=1)[0]
        
        return {
            'explanation': explanation,
            'prediction': predicted_class.cpu().detach().numpy(),
            'probabilities': probabilities.cpu().detach().numpy(),
            'confidence': confidence.cpu().detach().numpy(),
            'input_image': input_array
        }
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to model input format"""
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        
        # Transpose to CHW
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def _denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """Denormalize image for visualization"""
        # ImageNet denormalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Denormalize
        if len(image.shape) == 3 and image.shape[0] == 3:
            # CHW format
            image = np.transpose(image, (1, 2, 0))
        
        image = image * std + mean
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def visualize_shap_explanation(self, explanation: Dict, 
                                 save_path: Optional[Path] = None,
                                 title: str = "SHAP Explanation") -> None:
        """Visualize SHAP explanation"""
        
        shap_values = explanation['shap_values']
        input_data = explanation['input_data']
        
        # Handle batch input
        if len(shap_values.shape) == 4:
            shap_values = shap_values[0]
            input_data = input_data[0]
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if input_data.shape[0] == 3:
            original_img = np.transpose(input_data, (1, 2, 0))
        else:
            original_img = input_data
        
        original_img = self._denormalize_image(original_img)
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # SHAP values for fraud class
        if len(shap_values.shape) == 4:
            fraud_shap = shap_values[1]  # Fraud class
        else:
            fraud_shap = shap_values
        
        # Resize SHAP values to match image
        if fraud_shap.shape[0] == 3:
            fraud_shap = np.transpose(fraud_shap, (1, 2, 0))
        
        # Sum across channels for visualization
        fraud_shap_sum = np.sum(fraud_shap, axis=2)
        
        im = axes[1].imshow(fraud_shap_sum, cmap='RdBu_r')
        axes[1].set_title('SHAP Values (Fraud)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        axes[2].imshow(original_img)
        axes[2].imshow(fraud_shap_sum, cmap='RdBu_r', alpha=0.6)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_lime_explanation(self, explanation: Dict,
                                 save_path: Optional[Path] = None,
                                 title: str = "LIME Explanation") -> None:
        """Visualize LIME explanation"""
        
        lime_explanation = explanation['explanation']
        input_image = explanation['input_image']
        
        # Get explanation for fraud class
        fraud_class = 1
        temp, mask = lime_explanation.get_image_and_mask(
            fraud_class,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(input_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # LIME explanation
        axes[1].imshow(mask)
        axes[1].set_title('LIME Explanation (Fraud)')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(input_image)
        axes[2].imshow(mask, alpha=0.6)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"LIME visualization saved to {save_path}")
        
        plt.show()
    
    def generate_feature_importance_plot(self, explanation: Dict,
                                       save_path: Optional[Path] = None,
                                       top_k: int = 20) -> None:
        """Generate feature importance plot"""
        
        feature_importance = explanation['feature_importance']
        
        # Get top features
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        top_features = feature_importance[top_indices]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_features)
        plt.yticks(range(len(top_features)), [f'Feature {i}' for i in top_indices])
        plt.xlabel('Feature Importance (|SHAP Value|)')
        plt.title(f'Top {top_k} Most Important Features')
        plt.gca().invert_yaxis()
        
        # Color bars by importance
        colors = plt.cm.viridis(top_features / top_features.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def explain_batch(self, data_loader, num_samples: int = 10) -> List[Dict]:
        """Generate explanations for a batch of samples"""
        
        explanations = []
        
        for i, (data, target) in enumerate(data_loader):
            if i >= num_samples:
                break
            
            # SHAP explanation
            if self.shap_explainer is not None:
                shap_explanation = self.explain_with_shap(data)
                explanations.append({
                    'sample_idx': i,
                    'true_label': target.item(),
                    'shap_explanation': shap_explanation
                })
            
            # LIME explanation
            if self.lime_explainer is not None:
                lime_explanation = self.explain_with_lime(data)
                if 'lime_explanation' in explanations[-1]:
                    explanations[-1]['lime_explanation'] = lime_explanation
                else:
                    explanations.append({
                        'sample_idx': i,
                        'true_label': target.item(),
                        'lime_explanation': lime_explanation
                    })
        
        return explanations
    
    def create_explanation_report(self, explanations: List[Dict],
                                save_path: Optional[Path] = None) -> str:
        """Create comprehensive explanation report"""
        
        report = []
        report.append("# Fraud Detection Explanation Report\n")
        
        for exp in explanations:
            sample_idx = exp['sample_idx']
            true_label = exp['true_label']
            
            report.append(f"## Sample {sample_idx}\n")
            report.append(f"**True Label:** {'Fraud' if true_label == 1 else 'Non-Fraud'}\n")
            
            if 'shap_explanation' in exp:
                shap_exp = exp['shap_explanation']
                predicted_class = shap_exp['prediction'][0]
                confidence = shap_exp['confidence'][0]
                probabilities = shap_exp['probabilities'][0]
                
                report.append(f"**Predicted Class:** {'Fraud' if predicted_class == 1 else 'Non-Fraud'}\n")
                report.append(f"**Confidence:** {confidence:.3f}\n")
                report.append(f"**Probabilities:** Non-Fraud: {probabilities[0]:.3f}, Fraud: {probabilities[1]:.3f}\n")
                
                # Feature importance
                top_features = np.argsort(shap_exp['feature_importance'])[-5:][::-1]
                report.append("**Top 5 Important Features:**\n")
                for j, feat_idx in enumerate(top_features):
                    importance = shap_exp['feature_importance'][feat_idx]
                    report.append(f"{j+1}. Feature {feat_idx}: {importance:.4f}\n")
            
            report.append("\n---\n")
        
        report_text = "".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Explanation report saved to {save_path}")
        
        return report_text


def main():
    """Test the explainability module"""
    from training import FraudDetectionTrainer
    from data_preprocessing import DataPreprocessor
    from config import DATASET_CONFIG, MODELS_DIR
    
    # Load a trained model
    model_path = MODELS_DIR / "best_resnet50_fraud_detector.pth"
    
    if not model_path.exists():
        logger.error("No trained model found. Please train a model first.")
        return
    
    # Load model
    model = get_model('resnet50')
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize explainer
    explainer = FraudDetectionExplainer(model)
    
    # Create data loader
    preprocessor = DataPreprocessor()
    _, test_loader = preprocessor.create_data_loaders(
        DATASET_CONFIG["train_dir"],
        DATASET_CONFIG["test_dir"]
    )
    
    # Setup explainers
    # Get background data for SHAP
    background_data = []
    for data, _ in test_loader:
        background_data.append(data)
        if len(background_data) >= 10:  # Use 10 samples as background
            break
    background_data = torch.cat(background_data, dim=0)
    
    explainer.setup_shap_explainer(background_data, method='gradient')
    explainer.setup_lime_explainer()
    
    # Generate explanations for a few samples
    explanations = explainer.explain_batch(test_loader, num_samples=3)
    
    # Visualize explanations
    for i, exp in enumerate(explanations):
        if 'shap_explanation' in exp:
            explainer.visualize_shap_explanation(
                exp['shap_explanation'],
                save_path=RESULTS_DIR / f"shap_explanation_sample_{i}.png"
            )
        
        if 'lime_explanation' in exp:
            explainer.visualize_lime_explanation(
                exp['lime_explanation'],
                save_path=RESULTS_DIR / f"lime_explanation_sample_{i}.png"
            )
    
    # Create report
    report = explainer.create_explanation_report(
        explanations,
        save_path=RESULTS_DIR / "explanation_report.md"
    )
    
    print("Explanation report generated successfully!")


if __name__ == "__main__":
    main()
