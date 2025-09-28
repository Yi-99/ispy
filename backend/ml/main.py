"""
Main entry point for the Car Insurance Fraud Detection System
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from config import DATASET_CONFIG, MODELS_DIR, RESULTS_DIR
from models import get_model
from src import (
    DataPreprocessor, 
    EnhancedDataPreprocessor,
    FraudDetectionTrainer, 
    EnhancedFraudTrainer,
    HybridFraudDetector,
    FraudDetectionEvaluator,
    FraudDetectionExplainer,
    run_api
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_models(enhanced=False):
    """Train all fraud detection models"""
    logger.info("Starting model training...")
    
    if enhanced:
        logger.info("🚀 Using ENHANCED training pipeline with fraud-focused features!")
        # Initialize enhanced data preprocessor
        preprocessor = EnhancedDataPreprocessor(
            fraud_augmentation_factor=8,
            use_enhanced_augmentation=True
        )
        
        # Create enhanced data loaders
        train_loader, test_loader = preprocessor.create_data_loaders(
            DATASET_CONFIG["train_dir"],
            DATASET_CONFIG["test_dir"],
            use_enhanced=True
        )
        
        # Calculate class counts for enhanced training
        targets = [target for _, target in train_loader.dataset]
        class_counts = [targets.count(0), targets.count(1)]
        
        logger.info(f"Class distribution: Non-fraud: {class_counts[0]}, Fraud: {class_counts[1]}")
        
        # Train different models with enhanced pipeline
        model_names = ['resnet50', 'efficientnet', 'custom_cnn']
        results = {}
        
        for model_name in model_names:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name} with ENHANCED fraud detection")
            logger.info(f"{'='*50}")
            
            # Initialize enhanced trainer
            trainer = EnhancedFraudTrainer(model_name=model_name)
            
            # Train model with enhanced pipeline
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
        import json
        with open(RESULTS_DIR / "enhanced_training_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("🎉 Enhanced training completed! Results saved to results/ directory.")
        
    else:
        logger.info("Using standard training pipeline...")
        # Initialize data preprocessor
        preprocessor = DataPreprocessor()
        
        # Create data loaders
        train_loader, test_loader = preprocessor.create_data_loaders(
            DATASET_CONFIG["train_dir"],
            DATASET_CONFIG["test_dir"]
        )
        
        # Calculate class weights for imbalanced dataset
        from sklearn.utils.class_weight import compute_class_weight
        import torch
        import numpy as np
        
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(all_labels),
            y=all_labels
        )
        class_weights = torch.FloatTensor(class_weights)
        
        logger.info(f"Class weights: {class_weights}")
        
        # Train different models
        model_names = ['resnet50', 'efficientnet', 'custom_cnn']
        results = {}
        
        for model_name in model_names:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'='*50}")
            
            # Initialize trainer
            trainer = FraudDetectionTrainer(model_name=model_name)
            
            # Train model
            val_metrics = trainer.train(train_loader, test_loader, class_weights=class_weights)
            
            # Evaluate on test set
            test_metrics = trainer.evaluate_model(test_loader)
            
            # Store results
            results[model_name] = {
                'validation_metrics': val_metrics,
                'test_metrics': test_metrics
            }
            
            # Save plots
            trainer.plot_training_history(RESULTS_DIR / f"{model_name}_training_history.png")
            trainer.plot_confusion_matrix(test_loader, RESULTS_DIR / f"{model_name}_confusion_matrix.png")
            trainer.plot_roc_curve(test_loader, RESULTS_DIR / f"{model_name}_roc_curve.png")
        
        # Save results
        import json
        with open(RESULTS_DIR / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Training completed! Results saved to results/ directory.")


def train_hybrid():
    """Train hybrid models"""
    logger.info("Starting hybrid model training...")
    
    from hybrid_approach import main as train_hybrid_main
    train_hybrid_main()


def evaluate_models():
    """Evaluate trained models"""
    logger.info("Starting model evaluation...")
    
    # Initialize evaluator
    evaluator = FraudDetectionEvaluator()
    
    # Create data loader
    preprocessor = DataPreprocessor()
    _, test_loader = preprocessor.create_data_loaders(
        DATASET_CONFIG["train_dir"],
        DATASET_CONFIG["test_dir"]
    )
    
    # Load and evaluate each model
    model_names = ['resnet50', 'efficientnet', 'custom_cnn']
    model_results = {}
    
    for model_name in model_names:
        model_path = MODELS_DIR / f"best_{model_name}_fraud_detector.pth"
        
        if not model_path.exists():
            logger.warning(f"Model {model_name} not found, skipping...")
            continue
        
        # Load model
        model = get_model(model_name)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Evaluate
        metrics = evaluator.evaluate_model(test_loader, model)
        model_results[model_name] = metrics
        
        # Generate visualizations
        evaluator.plot_comprehensive_metrics(
            metrics, 
            RESULTS_DIR / f"{model_name}_evaluation_metrics.png"
        )
        
        # Generate report
        report = evaluator.generate_evaluation_report(
            metrics, 
            f"{model_name} Fraud Detector",
            RESULTS_DIR / f"{model_name}_evaluation_report.md"
        )
    
    # Compare models
    if len(model_results) > 1:
        comparison_df = evaluator.compare_models(
            model_results,
            RESULTS_DIR / "model_comparison.png"
        )
        comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    
    logger.info("Evaluation completed!")


def generate_explanations():
    """Generate explanations for model predictions"""
    logger.info("Starting explanation generation...")
    
    from explainability import main as explain_main
    explain_main()


def run_demo():
    """Run a demonstration of the fraud detection system"""
    logger.info("Running fraud detection demo...")
    
    # Initialize components
    preprocessor = DataPreprocessor()
    evaluator = FraudDetectionEvaluator()
    
    # Create data loader
    _, test_loader = preprocessor.create_data_loaders(
        DATASET_CONFIG["train_dir"],
        DATASET_CONFIG["test_dir"]
    )
    
    # Load best model
    model_name = 'resnet50'
    model_path = MODELS_DIR / f"best_{model_name}_fraud_detector.pth"
    
    if not model_path.exists():
        logger.error("No trained model found. Please train models first.")
        return
    
    model = get_model(model_name)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Run predictions on a few samples
    logger.info("Running predictions on test samples...")
    
    correct_predictions = 0
    total_predictions = 0
    fraud_detected = 0
    false_positives = 0
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= 10:  # Limit to 10 samples for demo
                break
            
            # Make prediction
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1)
            
            # Calculate metrics
            is_correct = (prediction == target).item()
            is_fraud = target.item() == 1
            predicted_fraud = prediction.item() == 1
            
            if is_correct:
                correct_predictions += 1
            if predicted_fraud:
                fraud_detected += 1
                if not is_fraud:
                    false_positives += 1
            
            total_predictions += 1
            
            # Log results
            logger.info(f"Sample {i+1}:")
            logger.info(f"  True Label: {'Fraud' if is_fraud else 'Non-Fraud'}")
            logger.info(f"  Predicted: {'Fraud' if predicted_fraud else 'Non-Fraud'}")
            logger.info(f"  Confidence: {probabilities.max().item():.3f}")
            logger.info(f"  Fraud Probability: {probabilities[0, 1].item():.3f}")
            logger.info(f"  Correct: {is_correct}")
            logger.info("")
    
    # Summary
    accuracy = correct_predictions / total_predictions
    false_positive_rate = false_positives / fraud_detected if fraud_detected > 0 else 0
    
    logger.info("Demo Results:")
    logger.info(f"  Accuracy: {accuracy:.3f}")
    logger.info(f"  Fraud Cases Detected: {fraud_detected}")
    logger.info(f"  False Positives: {false_positives}")
    logger.info(f"  False Positive Rate: {false_positive_rate:.3f}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Car Insurance Fraud Detection System")
    parser.add_argument(
        "command",
        choices=["train", "train-hybrid", "evaluate", "explain", "demo", "api"],
        help="Command to execute"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help="Model to use (for specific commands)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server"
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use enhanced training pipeline with fraud-focused features"
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    try:
        if args.command == "train":
            train_models(enhanced=args.enhanced)
        elif args.command == "train-hybrid":
            train_hybrid()
        elif args.command == "evaluate":
            evaluate_models()
        elif args.command == "explain":
            generate_explanations()
        elif args.command == "demo":
            run_demo()
        elif args.command == "api":
            logger.info(f"Starting API server on port {args.port}")
            run_api()
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
