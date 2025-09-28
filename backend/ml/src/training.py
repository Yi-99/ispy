"""
Enhanced Training Pipeline for Car Insurance Fraud Detection
Combines improved fraud detection techniques with comprehensive evaluation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import warnings
from collections import Counter
import torch.nn.functional as F

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models import get_model, UncertaintyEstimator
from config import MODEL_CONFIG, DATASET_CONFIG, MODELS_DIR, RESULTS_DIR
from .data_preprocessing import DataPreprocessor


class ImprovedFocalLoss(nn.Module):
    """
    Improved Focal Loss for handling severe class imbalance
    Focuses training on hard examples and down-weights easy negatives
    """
    def __init__(self, alpha=0.8, gamma=3.0, reduction='mean'):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha  # Higher alpha for fraud class
        self.gamma = gamma  # Higher gamma for more focus on hard examples
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate probabilities
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha weight
        alpha_t = torch.ones_like(targets, dtype=torch.float)
        alpha_t[targets == 1] = self.alpha  # Fraud class
        alpha_t[targets == 0] = 1 - self.alpha  # Non-fraud class
        
        # Calculate focal weight
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        # Apply focal loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss that considers effective number of samples
    """
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        super(ClassBalancedLoss, self).__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        weights = self.weights.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, weight=weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class EnhancedFraudTrainer:
    """
    Enhanced trainer with focus on fraud detection performance
    Combines best practices from both training approaches
    """
    
    def __init__(self, model_name: str = 'resnet50', device: str = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.sampler = None
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.fraud_detection_rates = []
        self.false_positive_rates = []
        self.probability_separations = []
        
        # Best model tracking
        self.best_fraud_score = 0.0
        self.best_fraud_detection_rate = 0.0
        self.best_model_state = None
        self.best_model_path = None
        
        logger.info(f"Using device: {self.device}")
    
    def setup_model(self, num_classes: int = 2, class_counts: Optional[List] = None):
        """Setup model, optimizer, and loss function with fraud detection focus"""
        
        # Initialize model
        self.model = get_model(self.model_name, num_classes=num_classes)
        self.model = self.model.to(self.device)
        
        # Setup optimizer with different learning rates for different parts
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name or 'head' in name or 'fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-4, 'weight_decay': 1e-4},
            {'params': classifier_params, 'lr': 5e-4, 'weight_decay': 1e-3}
        ])
        
        # Cosine annealing with warm restarts for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Setup loss function based on class counts
        if class_counts is not None:
            # Use Class-Balanced Loss for severe imbalance
            samples_per_class = [class_counts[0], class_counts[1]]
            self.criterion = ClassBalancedLoss(samples_per_class, beta=0.9999, gamma=3.0)
            logger.info(f"Using ClassBalancedLoss with samples per class: {samples_per_class}")
        else:
            # Fallback to Focal Loss
            self.criterion = ImprovedFocalLoss(alpha=0.85, gamma=3.5)
            logger.info("Using ImprovedFocalLoss")
        
        logger.info(f"Model setup complete. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_weighted_sampler(self, train_loader):
        """Setup weighted sampler for balanced batches"""
        targets = []
        for _, target in train_loader.dataset:
            targets.append(target)
        
        class_sample_count = np.array([len(np.where(np.array(targets) == t)[0]) for t in np.unique(targets)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in targets])
        
        self.sampler = WeightedRandomSampler(
            weights=samples_weight,
            num_samples=len(samples_weight),
            replacement=True
        )
        
        logger.info(f"Weighted sampler created with class weights: {weight}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """Enhanced training epoch with fraud-focused metrics"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        # Create new dataloader with weighted sampling
        weighted_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=self.sampler,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory
        )
        
        progress_bar = tqdm(weighted_loader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Collect predictions for metrics
            total_loss += loss.item()
            probabilities = F.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            
            # Update progress bar with fraud-specific metrics
            if batch_idx % 10 == 0:
                fraud_mask = np.array(all_targets) == 1
                if np.sum(fraud_mask) > 0:
                    avg_fraud_prob = np.mean(np.array(all_probabilities)[fraud_mask])
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Fraud_Prob': f'{avg_fraud_prob:.3f}'
                    })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(weighted_loader)
        metrics = self._calculate_fraud_metrics(all_targets, all_predictions, all_probabilities)
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Enhanced validation with fraud detection focus"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                probabilities = F.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_fraud_metrics(all_targets, all_predictions, all_probabilities)
        
        return avg_loss, metrics
    
    def _calculate_fraud_metrics(self, targets: List, predictions: List, probabilities: List) -> Dict:
        """Calculate comprehensive fraud detection metrics"""
        targets = np.array(targets)
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Basic confusion matrix
        cm = confusion_matrix(targets, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Fraud detection specific metrics
        fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for fraud
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_fraud = 2 * (precision * fraud_detection_rate) / (precision + fraud_detection_rate) if (precision + fraud_detection_rate) > 0 else 0
        
        # Overall accuracy
        accuracy = (targets == predictions).mean()
        
        # Probability analysis
        fraud_mask = targets == 1
        non_fraud_mask = targets == 0
        
        fraud_probs = probabilities[fraud_mask] if np.any(fraud_mask) else np.array([])
        non_fraud_probs = probabilities[non_fraud_mask] if np.any(non_fraud_mask) else np.array([])
        
        # Probability separation
        fraud_mean = np.mean(fraud_probs) if len(fraud_probs) > 0 else 0
        non_fraud_mean = np.mean(non_fraud_probs) if len(non_fraud_probs) > 0 else 0
        prob_separation = fraud_mean - non_fraud_mean
        
        # High confidence predictions
        high_conf_fraud = np.sum(probabilities > 0.7)
        high_conf_correct = np.sum((probabilities > 0.7) & (targets == 1))
        
        # AUC
        try:
            auc = roc_auc_score(targets, probabilities) if len(np.unique(targets)) > 1 else 0
        except:
            auc = 0
        
        return {
            'accuracy': accuracy,
            'fraud_detection_rate': fraud_detection_rate,
            'false_positive_rate': false_positive_rate,
            'precision': precision,
            'f1_fraud': f1_fraud,
            'auc': auc,
            'fraud_prob_mean': fraud_mean,
            'non_fraud_prob_mean': non_fraud_mean,
            'prob_separation': prob_separation,
            'high_conf_predictions': high_conf_fraud,
            'high_conf_correct': high_conf_correct,
            'confusion_matrix': cm.tolist()
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = None, class_counts: Optional[List] = None) -> Dict:
        """Enhanced training loop with fraud detection focus"""
        
        if epochs is None:
            epochs = MODEL_CONFIG.get("epochs", 100)
        
        # Calculate class counts if not provided
        if class_counts is None:
            targets = [target for _, target in train_loader.dataset]
            class_counts = [targets.count(0), targets.count(1)]
        
        # Setup model and weighted sampler
        self.setup_model(class_counts=class_counts)
        self.setup_weighted_sampler(train_loader)
        
        best_fraud_score = 0.0  # Combination of fraud detection rate and prob separation
        patience_counter = 0
        patience = MODEL_CONFIG.get("patience", 25)  # Higher patience for fraud detection
        
        logger.info(f"Starting enhanced training for {epochs} epochs...")
        logger.info(f"Class distribution: Non-fraud: {class_counts[0]}, Fraud: {class_counts[1]}")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.fraud_detection_rates.append(val_metrics['fraud_detection_rate'])
            self.false_positive_rates.append(val_metrics['false_positive_rate'])
            self.probability_separations.append(val_metrics['prob_separation'])
            
            # Log comprehensive metrics
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.3f}")
            logger.info(f"Fraud Detection Rate: {val_metrics['fraud_detection_rate']:.3f}")
            logger.info(f"False Positive Rate: {val_metrics['false_positive_rate']:.3f}")
            logger.info(f"Fraud Precision: {val_metrics['precision']:.3f}")
            logger.info(f"F1-Score (Fraud): {val_metrics['f1_fraud']:.3f}")
            logger.info(f"AUC: {val_metrics['auc']:.3f}")
            logger.info(f"Probability Separation: {val_metrics['prob_separation']:.4f}")
            logger.info(f"Fraud Prob Mean: {val_metrics['fraud_prob_mean']:.3f}")
            logger.info(f"Non-Fraud Prob Mean: {val_metrics['non_fraud_prob_mean']:.3f}")
            
            # Combined score for model selection (prioritize fraud detection)
            fraud_score = (val_metrics['fraud_detection_rate'] * 0.4 + 
                          val_metrics['f1_fraud'] * 0.3 + 
                          val_metrics['prob_separation'] * 0.2 + 
                          val_metrics['auc'] * 0.1)
            
            # Save best model based on fraud detection performance
            if fraud_score > best_fraud_score:
                best_fraud_score = fraud_score
                self.best_fraud_detection_rate = val_metrics['fraud_detection_rate']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save model
                self.best_model_path = MODELS_DIR / f"best_{self.model_name}_fraud_detector.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_metrics['accuracy'],
                    'val_metrics': val_metrics,
                    'model_name': self.model_name,
                    'fraud_score': fraud_score
                }, self.best_model_path)
                
                logger.info(f"New best fraud score: {fraud_score:.4f} (FDR: {val_metrics['fraud_detection_rate']:.3f})")
            else:
                patience_counter += 1
            
            # Early stopping with higher patience for fraud detection
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with fraud detection rate: {self.best_fraud_detection_rate:.3f}")
        
        return val_metrics
    
    def evaluate_with_thresholds(self, test_loader: DataLoader, thresholds=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]):
        """Comprehensive evaluation across different thresholds"""
        self.model.eval()
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)
        
        results = {}
        for threshold in thresholds:
            predictions = (all_probabilities >= threshold).astype(int)
            metrics = self._calculate_fraud_metrics(all_targets, predictions, all_probabilities)
            results[threshold] = metrics
        
        return results, all_probabilities, all_targets
    
    def plot_comprehensive_analysis(self, test_results, probabilities, targets, save_path=None):
        """Create comprehensive analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Training curves
        axes[0, 0].plot(self.fraud_detection_rates, label='Fraud Detection Rate', color='red')
        axes[0, 0].plot(self.false_positive_rates, label='False Positive Rate', color='blue')
        axes[0, 0].set_title('Fraud Detection Performance Over Training')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Threshold analysis
        thresholds = list(test_results.keys())
        fraud_detection_rates = [test_results[t]['fraud_detection_rate'] for t in thresholds]
        false_positive_rates = [test_results[t]['false_positive_rate'] for t in thresholds]
        
        axes[0, 1].plot(thresholds, fraud_detection_rates, 'o-', label='Fraud Detection Rate', color='red')
        axes[0, 1].plot(thresholds, false_positive_rates, 's-', label='False Positive Rate', color='blue')
        axes[0, 1].set_title('Performance vs Threshold')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Probability distributions
        fraud_probs = probabilities[targets == 1]
        non_fraud_probs = probabilities[targets == 0]
        
        axes[0, 2].hist(fraud_probs, bins=30, alpha=0.7, label='Fraud', color='red', density=True)
        axes[0, 2].hist(non_fraud_probs, bins=30, alpha=0.7, label='Non-Fraud', color='blue', density=True)
        axes[0, 2].set_title('Probability Distributions')
        axes[0, 2].set_xlabel('Fraud Probability')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. ROC Curve
        fpr, tpr, _ = roc_curve(targets, probabilities)
        auc = roc_auc_score(targets, probabilities)
        
        axes[1, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', color='darkblue')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 5. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        ap_score = average_precision_score(targets, probabilities)
        
        axes[1, 1].plot(recall, precision, label=f'PR Curve (AP = {ap_score:.3f})', color='green')
        axes[1, 1].set_title('Precision-Recall Curve')
        axes[1, 1].set_xlabel('Recall (Fraud Detection Rate)')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 6. Business Impact Analysis
        cost_missed = 1000
        cost_false_alarm = 50
        
        business_costs = []
        for threshold in thresholds:
            metrics = test_results[threshold]
            tp = np.sum((probabilities >= threshold) & (targets == 1))
            fp = np.sum((probabilities >= threshold) & (targets == 0))
            fn = np.sum((probabilities < threshold) & (targets == 1))
            
            total_cost = fn * cost_missed + fp * cost_false_alarm
            business_costs.append(total_cost)
        
        axes[1, 2].plot(thresholds, business_costs, 'o-', color='purple')
        axes[1, 2].set_title('Business Cost vs Threshold')
        axes[1, 2].set_xlabel('Threshold')
        axes[1, 2].set_ylabel('Total Cost ($)')
        axes[1, 2].grid(True)
        
        # Find optimal threshold
        optimal_idx = np.argmin(business_costs)
        optimal_threshold = thresholds[optimal_idx]
        axes[1, 2].axvline(optimal_threshold, color='red', linestyle='--', 
                          label=f'Optimal: {optimal_threshold:.2f}')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comprehensive analysis saved to {save_path}")
        
        plt.show()
        
        return optimal_threshold
    
    def generate_detailed_report(self, test_results, probabilities, targets):
        """Generate detailed fraud detection report"""
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE FRAUD DETECTION ANALYSIS REPORT")
        report.append("="*80)
        
        # Dataset overview
        total_samples = len(targets)
        fraud_cases = np.sum(targets == 1)
        non_fraud_cases = np.sum(targets == 0)
        
        report.append(f"\nDATASET OVERVIEW:")
        report.append(f"Total samples: {total_samples}")
        report.append(f"Fraud cases: {fraud_cases} ({fraud_cases/total_samples*100:.1f}%)")
        report.append(f"Non-fraud cases: {non_fraud_cases} ({non_fraud_cases/total_samples*100:.1f}%)")
        
        # Probability analysis
        fraud_probs = probabilities[targets == 1]
        non_fraud_probs = probabilities[targets == 0]
        
        report.append(f"\nPROBABILITY ANALYSIS:")
        report.append(f"Fraud cases - Mean: {np.mean(fraud_probs):.4f}, Std: {np.std(fraud_probs):.4f}")
        report.append(f"Non-fraud cases - Mean: {np.mean(non_fraud_probs):.4f}, Std: {np.std(non_fraud_probs):.4f}")
        report.append(f"Probability separation: {np.mean(fraud_probs) - np.mean(non_fraud_probs):.4f}")
        
        # High confidence analysis
        high_conf_fraud = np.sum(fraud_probs > 0.7)
        high_conf_non_fraud = np.sum(non_fraud_probs > 0.7)
        
        report.append(f"\nHIGH CONFIDENCE PREDICTIONS (>0.7):")
        report.append(f"Fraud cases with high confidence: {high_conf_fraud}/{len(fraud_probs)} ({high_conf_fraud/len(fraud_probs)*100:.1f}%)")
        report.append(f"Non-fraud cases with high confidence: {high_conf_non_fraud}/{len(non_fraud_probs)} ({high_conf_non_fraud/len(non_fraud_probs)*100:.1f}%)")
        
        # Threshold analysis
        report.append(f"\nTHRESHOLD ANALYSIS:")
        report.append(f"{'Threshold':<10} {'FDR':<8} {'FPR':<8} {'Precision':<10} {'F1':<8} {'Business Cost'}")
        report.append("-" * 65)
        
        best_threshold = None
        best_cost = float('inf')
        
        for threshold in sorted(test_results.keys()):
            metrics = test_results[threshold]
            
            # Calculate business cost
            tp = np.sum((probabilities >= threshold) & (targets == 1))
            fp = np.sum((probabilities >= threshold) & (targets == 0))
            fn = np.sum((probabilities < threshold) & (targets == 1))
            business_cost = fn * 1000 + fp * 50
            
            if business_cost < best_cost:
                best_cost = business_cost
                best_threshold = threshold
            
            report.append(f"{threshold:<10.2f} {metrics['fraud_detection_rate']:<8.3f} "
                         f"{metrics['false_positive_rate']:<8.3f} {metrics['precision']:<10.3f} "
                         f"{metrics['f1_fraud']:<8.3f} ${business_cost:<10,}")
        
        report.append(f"\nRECOMMENDED THRESHOLD: {best_threshold:.2f}")
        report.append(f"Expected business cost: ${best_cost:,}")
        
        # Model performance summary
        best_metrics = test_results[best_threshold]
        report.append(f"\nMODEL PERFORMANCE SUMMARY (at optimal threshold {best_threshold:.2f}):")
        report.append(f"Fraud Detection Rate: {best_metrics['fraud_detection_rate']:.3f}")
        report.append(f"False Positive Rate: {best_metrics['false_positive_rate']:.3f}")
        report.append(f"Precision: {best_metrics['precision']:.3f}")
        report.append(f"F1-Score: {best_metrics['f1_fraud']:.3f}")
        report.append(f"AUC: {best_metrics['auc']:.3f}")
        
        return "\n".join(report)
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Fraud detection metrics
        axes[0, 1].plot(self.fraud_detection_rates, label='Fraud Detection Rate', color='red')
        axes[0, 1].plot(self.false_positive_rates, label='False Positive Rate', color='blue')
        axes[0, 1].set_title('Fraud Detection Performance')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Probability separation
        axes[1, 0].plot(self.probability_separations, label='Probability Separation', color='green')
        axes[1, 0].set_title('Probability Separation Over Training')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Separation')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if hasattr(self.scheduler, 'get_last_lr'):
            lrs = [group['lr'] for group in self.optimizer.param_groups]
            axes[1, 1].plot(lrs)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()


def main():
    """Main training function with enhanced fraud detection"""
    # Initialize data preprocessor
    preprocessor = DataPreprocessor()
    
    # Create data loaders
    train_loader, test_loader = preprocessor.create_data_loaders(
        DATASET_CONFIG["train_dir"],
        DATASET_CONFIG["test_dir"]
    )
    
    # Calculate class counts for balanced training
    targets = [target for _, target in train_loader.dataset]
    class_counts = [targets.count(0), targets.count(1)]
    
    logger.info(f"Class distribution: Non-fraud: {class_counts[0]}, Fraud: {class_counts[1]}")
    
    # Train different models with enhanced pipeline
    model_names = ['resnet50', 'efficientnet', 'custom_cnn']
    results = {}
    
    for model_name in model_names:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name} with enhanced fraud detection")
        logger.info(f"{'='*50}")
        
        # Initialize enhanced trainer
        trainer = EnhancedFraudTrainer(model_name=model_name)
        
        # Train model
        val_metrics = trainer.train(train_loader, test_loader, class_counts=class_counts)
        
        # Comprehensive evaluation
        test_results, probabilities, targets = trainer.evaluate_with_thresholds(test_loader)
        
        # Generate analysis
        optimal_threshold = trainer.plot_comprehensive_analysis(
            test_results, probabilities, targets, 
            save_path=RESULTS_DIR / f"{model_name}_comprehensive_analysis.png"
        )
        
        # Generate report
        report = trainer.generate_detailed_report(test_results, probabilities, targets)
        print(report)
        
        # Save training history
        trainer.plot_training_history(RESULTS_DIR / f"{model_name}_training_history.png")
        
        # Store results
        results[model_name] = {
            'validation_metrics': val_metrics,
            'test_results': test_results,
            'optimal_threshold': optimal_threshold,
            'best_fraud_detection_rate': trainer.best_fraud_detection_rate
        }
    
    # Save results
    with open(RESULTS_DIR / "enhanced_training_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Enhanced training completed! Results saved to results/ directory.")
    
    return results


if __name__ == "__main__":
    main()