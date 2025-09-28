"""
Comprehensive evaluation framework for fraud detection models
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import RESULTS_DIR, FRAUD_CONFIG, DATASET_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionEvaluator:
    """Comprehensive evaluator for fraud detection models"""
    
    def __init__(self, model: nn.Module = None, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        if self.model:
            self.model.eval()
        
        logger.info("Fraud detection evaluator initialized")
    
    def evaluate_model(self, data_loader, model: nn.Module = None) -> Dict:
        """Comprehensive model evaluation"""
        
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model provided for evaluation")
        
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_uncertainties = []
        
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Evaluating model"):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                # Calculate uncertainty (entropy)
                uncertainty = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Fraud probability
                all_uncertainties.extend(uncertainty.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        uncertainties = np.array(all_uncertainties)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(targets, predictions, probabilities, uncertainties)
        
        return metrics
    
    def _calculate_comprehensive_metrics(self, targets: np.ndarray, predictions: np.ndarray,
                                       probabilities: np.ndarray, uncertainties: np.ndarray) -> Dict:
        """Calculate comprehensive fraud detection metrics"""
        
        # Basic classification metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions)
        recall = recall_score(targets, predictions)
        f1 = f1_score(targets, predictions)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Fraud-specific metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # ROC metrics
        try:
            auc_score = roc_auc_score(targets, probabilities)
            fpr, tpr, roc_thresholds = roc_curve(targets, probabilities)
        except:
            auc_score = 0.0
            fpr, tpr, roc_thresholds = np.array([]), np.array([]), np.array([])
        
        # Precision-Recall metrics
        try:
            precision_vals, recall_vals, pr_thresholds = precision_recall_curve(targets, probabilities)
            avg_precision = average_precision_score(targets, probabilities)
        except:
            precision_vals, recall_vals, pr_thresholds = np.array([]), np.array([]), np.array([])
            avg_precision = 0.0
        
        # Business metrics
        fraud_detection_rate = recall  # Same as recall for fraud class
        false_alarm_rate = false_positive_rate
        
        # Cost-based metrics (assuming cost of false negative > false positive)
        cost_false_negative = 1000  # Cost of missing fraud
        cost_false_positive = 100   # Cost of false alarm
        
        total_cost = (fn * cost_false_negative) + (fp * cost_false_positive)
        cost_per_prediction = total_cost / len(targets)
        
        # Uncertainty metrics
        uncertainty_mean = np.mean(uncertainties)
        uncertainty_std = np.std(uncertainties)
        
        # High uncertainty predictions
        uncertainty_threshold = np.percentile(uncertainties, 90)
        high_uncertainty_mask = uncertainties > uncertainty_threshold
        high_uncertainty_accuracy = accuracy_score(
            targets[high_uncertainty_mask], 
            predictions[high_uncertainty_mask]
        ) if np.sum(high_uncertainty_mask) > 0 else 0
        
        metrics = {
            # Basic metrics
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            
            # Fraud-specific metrics
            'fraud_detection_rate': fraud_detection_rate,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'false_alarm_rate': false_alarm_rate,
            
            # ROC/PR metrics
            'auc_score': auc_score,
            'average_precision': avg_precision,
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': roc_thresholds.tolist()},
            'pr_curve': {'precision': precision_vals.tolist(), 'recall': recall_vals.tolist(), 'thresholds': pr_thresholds.tolist()},
            
            # Business metrics
            'total_cost': total_cost,
            'cost_per_prediction': cost_per_prediction,
            'confusion_matrix': cm.tolist(),
            
            # Uncertainty metrics
            'uncertainty_mean': uncertainty_mean,
            'uncertainty_std': uncertainty_std,
            'high_uncertainty_accuracy': high_uncertainty_accuracy,
            'high_uncertainty_count': np.sum(high_uncertainty_mask),
            
            # Threshold analysis
            'optimal_threshold': self._find_optimal_threshold(targets, probabilities)
        }
        
        return metrics
    
    def _find_optimal_threshold(self, targets: np.ndarray, probabilities: np.ndarray) -> float:
        """Find optimal threshold based on F1 score"""
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            f1 = f1_score(targets, predictions)
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx]
    
    def plot_comprehensive_metrics(self, metrics: Dict, save_path: Optional[Path] = None):
        """Plot comprehensive evaluation metrics"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Confusion Matrix
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=DATASET_CONFIG["class_names"],
                   yticklabels=DATASET_CONFIG["class_names"],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        roc_data = metrics['roc_curve']
        if roc_data['fpr'] and roc_data['tpr']:
            axes[0, 1].plot(roc_data['fpr'], roc_data['tpr'], 
                           label=f'ROC Curve (AUC = {metrics["auc_score"]:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 3. Precision-Recall Curve
        pr_data = metrics['pr_curve']
        if pr_data['precision'] and pr_data['recall']:
            axes[0, 2].plot(pr_data['recall'], pr_data['precision'],
                           label=f'PR Curve (AP = {metrics["average_precision"]:.3f})')
            axes[0, 2].set_xlabel('Recall')
            axes[0, 2].set_ylabel('Precision')
            axes[0, 2].set_title('Precision-Recall Curve')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # 4. Key Metrics Bar Chart
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        metric_values = [metrics[metric] for metric in key_metrics]
        bars = axes[1, 0].bar(key_metrics, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
        axes[1, 0].set_title('Key Performance Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 5. Fraud Detection Metrics
        fraud_metrics = ['fraud_detection_rate', 'false_positive_rate', 'false_negative_rate']
        fraud_values = [metrics[metric] for metric in fraud_metrics]
        colors = ['green', 'red', 'orange']
        bars = axes[1, 1].bar(fraud_metrics, fraud_values, color=colors)
        axes[1, 1].set_title('Fraud Detection Metrics')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].set_ylim(0, 1)
        
        for bar, value in zip(bars, fraud_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Cost Analysis
        cost_metrics = ['total_cost', 'cost_per_prediction']
        cost_values = [metrics[metric] for metric in cost_metrics]
        axes[1, 2].bar(cost_metrics, cost_values, color=['red', 'orange'])
        axes[1, 2].set_title('Cost Analysis')
        axes[1, 2].set_ylabel('Cost')
        
        for i, value in enumerate(cost_values):
            axes[1, 2].text(i, value + max(cost_values) * 0.01,
                           f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comprehensive metrics plot saved to {save_path}")
        
        plt.show()
    
    def plot_calibration_curve(self, targets: np.ndarray, probabilities: np.ndarray,
                             save_path: Optional[Path] = None):
        """Plot calibration curve for probability calibration"""
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            targets, probabilities, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.ylabel('Fraction of positives')
        plt.xlabel('Mean predicted probability')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration curve saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, metrics: Dict, save_path: Optional[Path] = None):
        """Create interactive dashboard using Plotly"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve',
                          'Key Metrics', 'Fraud Detection Metrics', 'Cost Analysis'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}, {"type": "scatter"},
                   {"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Confusion Matrix
        cm = np.array(metrics['confusion_matrix'])
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=DATASET_CONFIG["class_names"],
                y=DATASET_CONFIG["class_names"],
                colorscale='Blues',
                showscale=True
            ),
            row=1, col=1
        )
        
        # 2. ROC Curve
        roc_data = metrics['roc_curve']
        if roc_data['fpr'] and roc_data['tpr']:
            fig.add_trace(
                go.Scatter(
                    x=roc_data['fpr'],
                    y=roc_data['tpr'],
                    mode='lines',
                    name=f'ROC (AUC={metrics["auc_score"]:.3f})',
                    line=dict(color='blue')
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=2
            )
        
        # 3. Precision-Recall Curve
        pr_data = metrics['pr_curve']
        if pr_data['precision'] and pr_data['recall']:
            fig.add_trace(
                go.Scatter(
                    x=pr_data['recall'],
                    y=pr_data['precision'],
                    mode='lines',
                    name=f'PR (AP={metrics["average_precision"]:.3f})',
                    line=dict(color='green')
                ),
                row=1, col=3
            )
        
        # 4. Key Metrics
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        metric_values = [metrics[metric] for metric in key_metrics]
        fig.add_trace(
            go.Bar(
                x=key_metrics,
                y=metric_values,
                name='Key Metrics',
                marker_color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
            ),
            row=2, col=1
        )
        
        # 5. Fraud Detection Metrics
        fraud_metrics = ['fraud_detection_rate', 'false_positive_rate', 'false_negative_rate']
        fraud_values = [metrics[metric] for metric in fraud_metrics]
        fig.add_trace(
            go.Bar(
                x=fraud_metrics,
                y=fraud_values,
                name='Fraud Metrics',
                marker_color=['green', 'red', 'orange']
            ),
            row=2, col=2
        )
        
        # 6. Cost Analysis
        cost_metrics = ['total_cost', 'cost_per_prediction']
        cost_values = [metrics[metric] for metric in cost_metrics]
        fig.add_trace(
            go.Bar(
                x=cost_metrics,
                y=cost_values,
                name='Cost Analysis',
                marker_color=['red', 'orange']
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text="Fraud Detection Model Evaluation Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        fig.show()
    
    def generate_evaluation_report(self, metrics: Dict, model_name: str = "Model",
                                 save_path: Optional[Path] = None) -> str:
        """Generate comprehensive evaluation report"""
        
        report = []
        report.append(f"# {model_name} Evaluation Report\n")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        report.append("## Executive Summary\n")
        report.append(f"- **Overall Accuracy:** {metrics['accuracy']:.3f}\n")
        report.append(f"- **Fraud Detection Rate:** {metrics['fraud_detection_rate']:.3f}\n")
        report.append(f"- **False Positive Rate:** {metrics['false_positive_rate']:.3f}\n")
        report.append(f"- **AUC Score:** {metrics['auc_score']:.3f}\n")
        report.append(f"- **Total Cost:** ${metrics['total_cost']:.0f}\n\n")
        
        # Detailed Metrics
        report.append("## Detailed Performance Metrics\n")
        report.append("### Classification Metrics\n")
        report.append(f"- Accuracy: {metrics['accuracy']:.4f}\n")
        report.append(f"- Precision: {metrics['precision']:.4f}\n")
        report.append(f"- Recall: {metrics['recall']:.4f}\n")
        report.append(f"- F1-Score: {metrics['f1_score']:.4f}\n")
        report.append(f"- Specificity: {metrics['specificity']:.4f}\n\n")
        
        report.append("### Fraud Detection Specific Metrics\n")
        report.append(f"- Fraud Detection Rate: {metrics['fraud_detection_rate']:.4f}\n")
        report.append(f"- False Positive Rate: {metrics['false_positive_rate']:.4f}\n")
        report.append(f"- False Negative Rate: {metrics['false_negative_rate']:.4f}\n")
        report.append(f"- False Alarm Rate: {metrics['false_alarm_rate']:.4f}\n\n")
        
        report.append("### ROC and PR Metrics\n")
        report.append(f"- AUC Score: {metrics['auc_score']:.4f}\n")
        report.append(f"- Average Precision: {metrics['average_precision']:.4f}\n")
        report.append(f"- Optimal Threshold: {metrics['optimal_threshold']:.4f}\n\n")
        
        report.append("### Business Impact Metrics\n")
        report.append(f"- Total Cost: ${metrics['total_cost']:.0f}\n")
        report.append(f"- Cost per Prediction: ${metrics['cost_per_prediction']:.2f}\n\n")
        
        report.append("### Uncertainty Analysis\n")
        report.append(f"- Mean Uncertainty: {metrics['uncertainty_mean']:.4f}\n")
        report.append(f"- Uncertainty Std: {metrics['uncertainty_std']:.4f}\n")
        report.append(f"- High Uncertainty Count: {metrics['high_uncertainty_count']}\n")
        report.append(f"- High Uncertainty Accuracy: {metrics['high_uncertainty_accuracy']:.4f}\n\n")
        
        # Confusion Matrix
        report.append("## Confusion Matrix\n")
        cm = np.array(metrics['confusion_matrix'])
        report.append("```\n")
        report.append(f"                Predicted\n")
        report.append(f"Actual    Non-Fraud    Fraud\n")
        report.append(f"Non-Fraud    {cm[0,0]:4d}      {cm[0,1]:4d}\n")
        report.append(f"Fraud        {cm[1,0]:4d}      {cm[1,1]:4d}\n")
        report.append("```\n\n")
        
        # Recommendations
        report.append("## Recommendations\n")
        
        if metrics['false_positive_rate'] > 0.1:
            report.append("- **High False Positive Rate:** Consider adjusting threshold or improving feature engineering\n")
        
        if metrics['false_negative_rate'] > 0.2:
            report.append("- **High False Negative Rate:** Critical - missing fraud cases. Consider ensemble methods or additional data\n")
        
        if metrics['auc_score'] < 0.8:
            report.append("- **Low AUC Score:** Model performance needs improvement. Consider feature engineering or different algorithms\n")
        
        if metrics['uncertainty_mean'] > 0.5:
            report.append("- **High Uncertainty:** Model confidence is low. Consider additional training or ensemble methods\n")
        
        report.append("\n---\n")
        report.append("*Report generated by Fraud Detection Evaluation Framework*\n")
        
        report_text = "".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report_text
    
    def compare_models(self, model_results: Dict[str, Dict], 
                      save_path: Optional[Path] = None) -> pd.DataFrame:
        """Compare multiple models"""
        
        comparison_data = []
        
        for model_name, metrics in model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'AUC': metrics['auc_score'],
                'Fraud Detection Rate': metrics['fraud_detection_rate'],
                'False Positive Rate': metrics['false_positive_rate'],
                'Total Cost': metrics['total_cost']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Key metrics comparison
        key_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        comparison_df.set_index('Model')[key_metrics].plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Key Metrics Comparison')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Fraud detection metrics
        fraud_metrics = ['Fraud Detection Rate', 'False Positive Rate']
        comparison_df.set_index('Model')[fraud_metrics].plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Fraud Detection Metrics')
        axes[0, 1].set_ylabel('Rate')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Cost comparison
        comparison_df.set_index('Model')['Total Cost'].plot(kind='bar', ax=axes[1, 0], color='red')
        axes[1, 0].set_title('Total Cost Comparison')
        axes[1, 0].set_ylabel('Cost')
        
        # ROC comparison (if available)
        axes[1, 1].text(0.5, 0.5, 'ROC Curves\n(Implementation needed)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('ROC Curves Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
        
        return comparison_df


def main():
    """Test the evaluation framework"""
    from training import FraudDetectionTrainer
    from data_preprocessing import DataPreprocessor
    from config import DATASET_CONFIG, MODELS_DIR
    
    # Initialize evaluator
    evaluator = FraudDetectionEvaluator()
    
    # Create data loader
    preprocessor = DataPreprocessor()
    _, test_loader = preprocessor.create_data_loaders(
        DATASET_CONFIG["train_dir"],
        DATASET_CONFIG["test_dir"]
    )
    
    # Test with a dummy model (in practice, load a trained model)
    from models import get_model
    model = get_model('resnet50')
    
    # Evaluate model
    metrics = evaluator.evaluate_model(test_loader, model)
    
    # Generate visualizations
    evaluator.plot_comprehensive_metrics(metrics, RESULTS_DIR / "evaluation_metrics.png")
    
    # Create interactive dashboard
    evaluator.create_interactive_dashboard(metrics, RESULTS_DIR / "evaluation_dashboard.html")
    
    # Generate report
    report = evaluator.generate_evaluation_report(metrics, "ResNet50 Fraud Detector", 
                                                RESULTS_DIR / "evaluation_report.md")
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
