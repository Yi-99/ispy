"""
Hybrid approach combining CNN feature extraction with traditional ML classifiers
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import MODELS_DIR, RESULTS_DIR
from models.resnet_model import ResNetFraudDetector
from models.efficientnet_model import EfficientNetFraudDetector
from .data_preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridFraudDetector:
    """Hybrid approach combining CNN features with traditional ML"""
    
    def __init__(self, cnn_model_name: str = 'resnet50', device: str = None):
        self.cnn_model_name = cnn_model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn_model = None
        self.feature_extractor = None
        self.traditional_models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
        logger.info(f"Initializing hybrid detector with {cnn_model_name} backbone")
    
    def setup_cnn_feature_extractor(self, num_classes: int = 2):
        """Setup CNN model for feature extraction"""
        
        # Load pretrained model
        if self.cnn_model_name == 'resnet50':
            self.cnn_model = ResNetFraudDetector(num_classes=num_classes)
        elif self.cnn_model_name == 'efficientnet':
            self.cnn_model = EfficientNetFraudDetector(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported CNN model: {self.cnn_model_name}")
        
        self.cnn_model = self.cnn_model.to(self.device)
        
        # Create feature extractor (remove final classification layer)
        if self.cnn_model_name == 'resnet50':
            self.feature_extractor = nn.Sequential(*list(self.cnn_model.children())[:-1])
        else:
            # For other models, we'll extract features from the penultimate layer
            self.feature_extractor = self.cnn_model.backbone
        
        self.feature_extractor.eval()
        logger.info("CNN feature extractor setup complete")
    
    def extract_cnn_features(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract CNN features from dataset"""
        self.feature_extractor.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Extracting CNN features"):
                data = data.to(self.device)
                
                # Extract features
                if self.cnn_model_name == 'resnet50':
                    cnn_features = self.feature_extractor(data)
                    cnn_features = cnn_features.view(cnn_features.size(0), -1)
                else:
                    # For other models, extract from backbone
                    cnn_features = self.feature_extractor.forward_features(data)
                    if hasattr(cnn_features, 'view'):
                        cnn_features = cnn_features.view(cnn_features.size(0), -1)
                
                features.extend(cnn_features.cpu().numpy())
                labels.extend(target.numpy())
        
        return np.array(features), np.array(labels)
    
    def extract_traditional_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract traditional computer vision features"""
        preprocessor = DataPreprocessor()
        features = preprocessor.extract_features(image_paths)
        return features
    
    def combine_features(self, cnn_features: np.ndarray, 
                        traditional_features: np.ndarray) -> np.ndarray:
        """Combine CNN and traditional features"""
        combined_features = np.concatenate([cnn_features, traditional_features], axis=1)
        
        # Create feature names
        cnn_feature_names = [f"cnn_feature_{i}" for i in range(cnn_features.shape[1])]
        traditional_feature_names = [f"traditional_feature_{i}" for i in range(traditional_features.shape[1])]
        self.feature_names = cnn_feature_names + traditional_feature_names
        
        logger.info(f"Combined features shape: {combined_features.shape}")
        return combined_features
    
    def train_traditional_models(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train multiple traditional ML models"""
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Define models to train
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                class_weight='balanced'
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val_scaled)
                y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                
                # Calculate metrics
                accuracy = (y_pred == y_val).mean()
                auc_score = roc_auc_score(y_val, y_pred_proba)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Store trained models
        self.traditional_models = {name: results[name]['model'] for name, model in results.items() 
                                 if 'model' in results[name]}
        
        return results
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Perform hyperparameter tuning for best models"""
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Define parameter grids
        param_grids = {
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [4, 6, 8]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [4, 6, 8]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        tuned_models = {}
        
        for name, param_grid in param_grids.items():
            logger.info(f"Tuning hyperparameters for {name}...")
            
            if name == 'xgboost':
                base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            elif name == 'lightgbm':
                base_model = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
            elif name == 'random_forest':
                base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
            
            # Grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            tuned_models[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
            
            logger.info(f"{name} best params: {grid_search.best_params_}")
            logger.info(f"{name} best CV score: {grid_search.best_score_:.4f}")
        
        return tuned_models
    
    def create_ensemble(self, models: Dict, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Create ensemble of best performing models"""
        
        X_val_scaled = self.scaler.transform(X_val)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model_info in models.items():
            if 'model' in model_info:
                model = model_info['model']
                pred = model.predict(X_val_scaled)
                proba = model.predict_proba(X_val_scaled)[:, 1]
                
                predictions[name] = pred
                probabilities[name] = proba
        
        # Simple voting ensemble
        if len(predictions) > 0:
            # Average probabilities
            ensemble_proba = np.mean(list(probabilities.values()), axis=0)
            ensemble_pred = (ensemble_proba > 0.5).astype(int)
            
            # Calculate ensemble metrics
            ensemble_accuracy = (ensemble_pred == y_val).mean()
            ensemble_auc = roc_auc_score(y_val, ensemble_proba)
            
            ensemble_results = {
                'predictions': ensemble_pred,
                'probabilities': ensemble_proba,
                'accuracy': ensemble_accuracy,
                'auc_score': ensemble_auc
            }
            
            logger.info(f"Ensemble - Accuracy: {ensemble_accuracy:.4f}, AUC: {ensemble_auc:.4f}")
            
            return ensemble_results
        
        return {}
    
    def save_models(self, save_dir: Path = None):
        """Save trained models"""
        if save_dir is None:
            save_dir = MODELS_DIR
        
        # Save scaler
        joblib.dump(self.scaler, save_dir / "hybrid_scaler.pkl")
        
        # Save traditional models
        for name, model in self.traditional_models.items():
            joblib.dump(model, save_dir / f"hybrid_{name}.pkl")
        
        # Save CNN model if available
        if self.cnn_model is not None:
            torch.save(self.cnn_model.state_dict(), save_dir / f"hybrid_{self.cnn_model_name}_cnn.pth")
        
        logger.info(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: Path = None):
        """Load trained models"""
        if save_dir is None:
            save_dir = MODELS_DIR
        
        # Load scaler
        self.scaler = joblib.load(save_dir / "hybrid_scaler.pkl")
        
        # Load traditional models
        for model_file in save_dir.glob("hybrid_*.pkl"):
            if model_file.name != "hybrid_scaler.pkl":
                name = model_file.stem.replace("hybrid_", "")
                self.traditional_models[name] = joblib.load(model_file)
        
        logger.info("Models loaded successfully")
    
    def predict(self, X: np.ndarray, model_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using trained models"""
        
        X_scaled = self.scaler.transform(X)
        
        if model_name and model_name in self.traditional_models:
            # Use specific model
            model = self.traditional_models[model_name]
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]
        else:
            # Use ensemble
            all_probs = []
            for model in self.traditional_models.values():
                prob = model.predict_proba(X_scaled)[:, 1]
                all_probs.append(prob)
            
            probabilities = np.mean(all_probs, axis=0)
            predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities


def main():
    """Main function to train hybrid models"""
    from data_preprocessing import DataPreprocessor
    from config import DATASET_CONFIG
    
    # Initialize components
    preprocessor = DataPreprocessor()
    hybrid_detector = HybridFraudDetector(cnn_model_name='resnet50')
    
    # Create data loaders
    train_loader, test_loader = preprocessor.create_data_loaders(
        DATASET_CONFIG["train_dir"],
        DATASET_CONFIG["test_dir"]
    )
    
    # Setup CNN feature extractor
    hybrid_detector.setup_cnn_feature_extractor()
    
    # Extract CNN features
    logger.info("Extracting CNN features...")
    train_cnn_features, train_labels = hybrid_detector.extract_cnn_features(train_loader)
    test_cnn_features, test_labels = hybrid_detector.extract_cnn_features(test_loader)
    
    # Extract traditional features
    logger.info("Extracting traditional features...")
    train_paths = []
    test_paths = []
    
    # Get image paths (simplified - in practice, you'd store these during data loading)
    for class_name in DATASET_CONFIG["class_names"]:
        train_class_dir = DATASET_CONFIG["train_dir"] / class_name
        test_class_dir = DATASET_CONFIG["test_dir"] / class_name
        
        if train_class_dir.exists():
            train_paths.extend(list(train_class_dir.glob("*.jpg")))
        if test_class_dir.exists():
            test_paths.extend(list(test_class_dir.glob("*.jpg")))
    
    train_traditional_features = hybrid_detector.extract_traditional_features([str(p) for p in train_paths])
    test_traditional_features = hybrid_detector.extract_traditional_features([str(p) for p in test_paths])
    
    # Combine features
    X_train = hybrid_detector.combine_features(train_cnn_features, train_traditional_features)
    X_test = hybrid_detector.combine_features(test_cnn_features, test_traditional_features)
    
    # Split training data for validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    # Train traditional models
    logger.info("Training traditional ML models...")
    model_results = hybrid_detector.train_traditional_models(
        X_train_split, y_train_split, X_val_split, y_val_split
    )
    
    # Hyperparameter tuning
    logger.info("Performing hyperparameter tuning...")
    tuned_models = hybrid_detector.hyperparameter_tuning(X_train_split, y_train_split)
    
    # Create ensemble
    logger.info("Creating ensemble...")
    ensemble_results = hybrid_detector.create_ensemble(model_results, X_val_split, y_val_split)
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_predictions, test_probabilities = hybrid_detector.predict(X_test)
    
    test_accuracy = (test_predictions == test_labels).mean()
    test_auc = roc_auc_score(test_labels, test_probabilities)
    
    logger.info(f"Final Test Results:")
    logger.info(f"Accuracy: {test_accuracy:.4f}")
    logger.info(f"AUC: {test_auc:.4f}")
    
    # Save models
    hybrid_detector.save_models()
    
    # Save results
    results = {
        'model_results': model_results,
        'tuned_models': tuned_models,
        'ensemble_results': ensemble_results,
        'test_accuracy': test_accuracy,
        'test_auc': test_auc
    }
    
    import json
    with open(RESULTS_DIR / "hybrid_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Hybrid training completed!")


if __name__ == "__main__":
    main()
