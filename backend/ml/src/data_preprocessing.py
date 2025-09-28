"""
Enhanced Data Preprocessing Pipeline for Car Insurance Fraud Detection
Combines standard and advanced fraud-specific preprocessing techniques
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageEnhance, ImageFilter
import logging
import random
from collections import Counter
import imgaug.augmenters as iaa

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import DATASET_CONFIG, PREPROCESSING_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedCarFraudDataset(Dataset):
    """
    Enhanced dataset with fraud-specific augmentation strategies
    Combines standard and aggressive augmentation techniques
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, fraud_transform=None, is_training: bool = True,
                 fraud_augmentation_factor: int = 5, use_enhanced_augmentation: bool = True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.fraud_transform = fraud_transform
        self.is_training = is_training
        self.fraud_augmentation_factor = fraud_augmentation_factor
        self.use_enhanced_augmentation = use_enhanced_augmentation
        
        # Create augmented fraud samples during training if enhanced augmentation is enabled
        if is_training and use_enhanced_augmentation:
            self._create_augmented_fraud_samples()
        
    def _create_augmented_fraud_samples(self):
        """Create additional augmented samples for fraud cases"""
        fraud_indices = [i for i, label in enumerate(self.labels) if label == 1]
        
        augmented_paths = []
        augmented_labels = []
        
        for _ in range(self.fraud_augmentation_factor):
            for idx in fraud_indices:
                augmented_paths.append(self.image_paths[idx])
                augmented_labels.append(1)
        
        # Add augmented samples
        self.image_paths.extend(augmented_paths)
        self.labels.extend(augmented_labels)
        
        logger.info(f"Added {len(augmented_paths)} augmented fraud samples")
        logger.info(f"New dataset size: {len(self.image_paths)}")
        logger.info(f"New class distribution: {Counter(self.labels)}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply class-specific transforms
        if self.is_training and label == 1 and self.fraud_transform and self.use_enhanced_augmentation:
            # Use aggressive augmentation for fraud cases
            image = self.fraud_transform(image=image)["image"]
        elif self.transform:
            # Standard augmentation for non-fraud or inference
            image = self.transform(image=image)["image"]
        
        return image, label


class EnhancedDataPreprocessor:
    """
    Enhanced data preprocessor with fraud-specific strategies
    Combines standard and advanced preprocessing techniques
    """
    
    def __init__(self, fraud_augmentation_factor: int = 5, use_enhanced_augmentation: bool = True):
        self.scaler = StandardScaler()
        self.fraud_augmentation_factor = fraud_augmentation_factor
        self.use_enhanced_augmentation = use_enhanced_augmentation
        
        # Initialize transforms
        self.train_transform = self._get_train_transform()
        self.fraud_transform = self._get_fraud_transform() if use_enhanced_augmentation else None
        self.val_transform = self._get_val_transform()
        
    def _get_train_transform(self):
        """Standard training augmentation with improved parameters"""
        return A.Compose([
            A.Resize(*PREPROCESSING_CONFIG["resize"]),
            A.RandomCrop(224, 224),
            A.OneOf([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=45, p=0.5),  # Increased from 20 to 45
            ], p=0.8),  # Increased from 0.5 to 0.8
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),  # Added vertical flip
            ], p=0.8),  # Increased from 0.5 to 0.8
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,  # Increased from [0.8, 1.2] to 0.3
                    contrast_limit=0.3,    # Increased from [0.8, 1.2] to 0.3
                    p=0.8
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=0.8),  # Increased range
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            ], p=0.8),  # Increased from 0.5 to 0.8
            A.OneOf([
                A.GaussNoise(var_limit=(20.0, 80.0), p=0.5),  # Increased noise
                A.GaussianBlur(blur_limit=(5, 9), p=0.5),     # Increased blur
                A.MotionBlur(blur_limit=7, p=0.5),
            ], p=0.5),  # Keep at 0.5
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _get_fraud_transform(self):
        """Aggressive augmentation specifically for fraud cases"""
        return A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            
            # More aggressive geometric transforms
            A.OneOf([
                A.RandomRotate90(p=0.7),
                A.Rotate(limit=45, p=0.7),
                A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), 
                        rotate=(-45, 45), shear=(-10, 10), p=0.6),
            ], p=0.9),
            
            # Flip operations
            A.OneOf([
                A.HorizontalFlip(p=0.8),
                A.VerticalFlip(p=0.4),
                A.Transpose(p=0.3),
            ], p=0.8),
            
            # Color and lighting augmentations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
                A.RandomGamma(gamma_limit=(60, 140), p=0.8),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.7),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
            ], p=0.9),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(20.0, 100.0), p=0.6),
                A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4),
            ], p=0.6),
            
            A.OneOf([
                A.GaussianBlur(blur_limit=(5, 11), p=0.6),
                A.MotionBlur(blur_limit=11, p=0.6),
                A.MedianBlur(blur_limit=7, p=0.3),
            ], p=0.5),
            
            # Weather and environmental effects
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.3),
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, 
                           drop_width=1, drop_color=(200, 200, 200), p=0.2),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, 
                               angle_upper=1, num_flare_circles_lower=6, 
                               num_flare_circles_upper=10, p=0.1),
            ], p=0.3),
            
            # Distortion effects
            A.OneOf([
                A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
            ], p=0.4),
            
            # Quality degradation
            A.OneOf([
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.4),
                A.Downscale(scale_min=0.7, scale_max=0.9, p=0.3),
            ], p=0.3),
            
            # Cutout and dropout
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                              min_holes=1, min_height=8, min_width=8, p=0.4),
                A.CoarseDropout(max_holes=16, max_height=16, max_width=16, 
                              min_holes=1, min_height=4, min_width=4, p=0.3),
            ], p=0.3),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _get_val_transform(self):
        """Validation/test transforms"""
        return A.Compose([
            A.Resize(*PREPROCESSING_CONFIG["resize"]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def create_balanced_sampler(self, dataset):
        """Create weighted sampler for balanced training"""
        targets = [dataset[i][1] for i in range(len(dataset))]
        class_counts = Counter(targets)
        
        # Calculate weights inversely proportional to class frequency
        total_samples = len(targets)
        class_weights = {}
        for class_id, count in class_counts.items():
            class_weights[class_id] = total_samples / (len(class_counts) * count)
        
        # Create sample weights
        sample_weights = [class_weights[target] for target in targets]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def load_dataset(self, data_dir: Path) -> Tuple[List[str], List[int]]:
        """Load dataset from directory structure"""
        image_paths = []
        labels = []
        
        for class_name in DATASET_CONFIG["class_names"]:
            class_dir = data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directory {class_dir} does not exist")
                continue
                
            class_label = DATASET_CONFIG["fraud_class"] if class_name == "Fraud" else DATASET_CONFIG["non_fraud_class"]
            
            # Support multiple image formats
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
            
            for ext in image_extensions:
                for image_file in class_dir.glob(ext):
                    image_paths.append(str(image_file))
                    labels.append(class_label)
        
        logger.info(f"Loaded {len(image_paths)} images from {data_dir}")
        logger.info(f"Class distribution: {Counter(labels)}")
        
        return image_paths, labels
    
    def create_data_loaders(self, train_dir: Path, test_dir: Path, 
                          batch_size: int = None, use_enhanced: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Create training and test data loaders with optional enhanced features"""
        if batch_size is None:
            batch_size = DATASET_CONFIG["batch_size"]
        
        # Load training data
        train_paths, train_labels = self.load_dataset(train_dir)
        train_dataset = EnhancedCarFraudDataset(
            train_paths, train_labels,
            transform=self.train_transform,
            fraud_transform=self.fraud_transform,
            is_training=True,
            fraud_augmentation_factor=self.fraud_augmentation_factor,
            use_enhanced_augmentation=use_enhanced
        )
        
        # Load test data
        test_paths, test_labels = self.load_dataset(test_dir)
        test_dataset = EnhancedCarFraudDataset(
            test_paths, test_labels,
            transform=self.val_transform,
            fraud_transform=None,
            is_training=False,
            use_enhanced_augmentation=False
        )
        
        # Create data loaders
        if use_enhanced and self.use_enhanced_augmentation:
            # Use balanced sampler for enhanced training
            sampler = self.create_balanced_sampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=DATASET_CONFIG["num_workers"],
                pin_memory=True,
                drop_last=True
            )
        else:
            # Standard training loader
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=DATASET_CONFIG["num_workers"],
                pin_memory=True
            )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=DATASET_CONFIG["num_workers"],
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def analyze_dataset_quality(self, data_dir: Path):
        """Analyze dataset quality and provide recommendations"""
        image_paths, labels = self.load_dataset(data_dir)
        
        analysis = {
            'total_images': len(image_paths),
            'class_distribution': Counter(labels),
            'image_qualities': [],
            'corrupted_images': [],
            'recommendations': []
        }
        
        # Sample images for quality analysis
        sample_size = min(100, len(image_paths))
        sample_indices = random.sample(range(len(image_paths)), sample_size)
        
        for idx in sample_indices:
            try:
                image = cv2.imread(image_paths[idx])
                if image is None:
                    analysis['corrupted_images'].append(image_paths[idx])
                    continue
                
                # Calculate image quality metrics
                height, width = image.shape[:2]
                blur_score = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                brightness = np.mean(image)
                
                analysis['image_qualities'].append({
                    'path': image_paths[idx],
                    'label': labels[idx],
                    'size': (width, height),
                    'blur_score': blur_score,
                    'brightness': brightness
                })
                
            except Exception as e:
                analysis['corrupted_images'].append(image_paths[idx])
        
        # Generate recommendations
        fraud_count = analysis['class_distribution'].get(1, 0)
        non_fraud_count = analysis['class_distribution'].get(0, 0)
        imbalance_ratio = non_fraud_count / fraud_count if fraud_count > 0 else float('inf')
        
        if imbalance_ratio > 20:
            analysis['recommendations'].append(f"Severe class imbalance detected ({imbalance_ratio:.1f}:1). Consider collecting more fraud samples.")
        
        if analysis['image_qualities']:
            avg_blur = np.mean([q['blur_score'] for q in analysis['image_qualities']])
            if avg_blur < 100:
                analysis['recommendations'].append("Low image quality detected. Consider image enhancement techniques.")
        
        if len(analysis['corrupted_images']) > 0:
            analysis['recommendations'].append(f"Found {len(analysis['corrupted_images'])} corrupted images. Clean dataset recommended.")
        
        return analysis
    
    def extract_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract traditional computer vision features"""
        features = []
        
        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize image
                image = cv2.resize(image, (224, 224))
                
                # Extract features
                feature_vector = self._extract_cv_features(image)
                features.append(feature_vector)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                # Add zero features as fallback
                features.append(np.zeros(1000))  # Adjust size based on feature extraction
        
        return np.array(features)
    
    def _extract_cv_features(self, image: np.ndarray) -> np.ndarray:
        """Extract computer vision features from image"""
        features = []
        
        # Color features
        mean_color = np.mean(image, axis=(0, 1))
        std_color = np.std(image, axis=(0, 1))
        features.extend(mean_color)
        features.extend(std_color)
        
        # Texture features using Local Binary Patterns
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features.extend(hist.flatten()[:100])  # Use first 100 bins
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # Texture features using GLCM (simplified)
        texture_features = self._simple_texture_features(gray)
        features.extend(texture_features)
        
        # Ensure consistent feature size
        while len(features) < 1000:
            features.append(0.0)
        
        return np.array(features[:1000])
    
    def _simple_texture_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract simple texture features"""
        features = []
        
        # Standard deviation of pixel values
        features.append(np.std(gray_image))
        
        # Mean of pixel values
        features.append(np.mean(gray_image))
        
        # Gradient features
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))
        
        # Fill remaining features with zeros
        while len(features) < 50:
            features.append(0.0)
        
        return features[:50]
    
    def get_class_weights(self, labels: List[int]) -> Dict[int, float]:
        """Calculate class weights for imbalanced dataset"""
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        # Calculate weights inversely proportional to class frequency
        weights = {}
        for class_id, count in class_counts.items():
            weights[class_id] = total_samples / (len(class_counts) * count)
        
        logger.info(f"Class weights: {weights}")
        return weights
    
    def create_mixup_loader(self, train_loader, alpha=0.4):
        """Create MixUp augmented data loader for better generalization"""
        class MixUpDataLoader:
            def __init__(self, loader, alpha):
                self.loader = loader
                self.alpha = alpha
                
            def __iter__(self):
                for batch_idx, (data, target) in enumerate(self.loader):
                    if self.alpha > 0:
                        lam = np.random.beta(self.alpha, self.alpha)
                        batch_size = data.size(0)
                        index = torch.randperm(batch_size)
                        
                        mixed_data = lam * data + (1 - lam) * data[index, :]
                        target_a, target_b = target, target[index]
                        
                        yield mixed_data, target_a, target_b, lam
                    else:
                        yield data, target, target, 1.0
            
            def __len__(self):
                return len(self.loader)
        
        return MixUpDataLoader(train_loader, alpha)
    
    def create_fraud_specific_splits(self, image_paths, labels, test_size=0.2, random_state=42):
        """Create stratified splits ensuring fraud cases are well represented in both sets"""
        # Separate fraud and non-fraud cases
        fraud_indices = [i for i, label in enumerate(labels) if label == 1]
        non_fraud_indices = [i for i, label in enumerate(labels) if label == 0]
        
        # Ensure minimum fraud cases in test set
        min_fraud_test = max(10, int(len(fraud_indices) * test_size))
        min_fraud_train = len(fraud_indices) - min_fraud_test
        
        # Split fraud cases
        fraud_train_indices = fraud_indices[:min_fraud_train]
        fraud_test_indices = fraud_indices[min_fraud_train:]
        
        # Split non-fraud cases proportionally
        non_fraud_test_size = len(fraud_test_indices) * (len(non_fraud_indices) // len(fraud_indices))
        non_fraud_train_indices = non_fraud_indices[:-non_fraud_test_size]
        non_fraud_test_indices = non_fraud_indices[-non_fraud_test_size:]
        
        # Combine indices
        train_indices = fraud_train_indices + non_fraud_train_indices
        test_indices = fraud_test_indices + non_fraud_test_indices
        
        # Create splits
        train_paths = [image_paths[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_paths = [image_paths[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        
        logger.info(f"Train split: {Counter(train_labels)}")
        logger.info(f"Test split: {Counter(test_labels)}")
        
        return train_paths, train_labels, test_paths, test_labels


def apply_advanced_fraud_augmentation(image):
    """
    Apply advanced augmentation techniques specifically for fraud detection
    """
    # Convert to PIL for some operations
    pil_image = Image.fromarray(image)
    
    # Random combination of techniques
    techniques = []
    
    # 1. Perspective transformation (simulates different camera angles)
    if random.random() < 0.4:
        techniques.append(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
    
    # 2. Lighting variations (important for fraud detection)
    if random.random() < 0.6:
        techniques.extend([
            iaa.MultiplyAndAddToBrightness(mul=(0.7, 1.3), add=(-20, 20)),
            iaa.GammaContrast((0.7, 1.3))
        ])
    
    # 3. Weather simulation
    if random.random() < 0.3:
        techniques.extend([
            iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.1, 0.3))
        ])
    
    # 4. Quality degradation (simulates different device qualities)
    if random.random() < 0.4:
        techniques.extend([
            iaa.JpegCompression(compression=(70, 99)),
            iaa.Resize({"height": (0.7, 1.0), "width": (0.7, 1.0)})
        ])
    
    # Apply techniques
    if techniques:
        seq = iaa.Sequential(techniques, random_order=True)
        augmented = seq(image=image)
        return augmented
    
    return image


# Legacy compatibility - maintain original class name
class DataPreprocessor(EnhancedDataPreprocessor):
    """Legacy compatibility wrapper for EnhancedDataPreprocessor"""
    
    def __init__(self, fraud_augmentation_factor: int = 5, use_enhanced_augmentation: bool = False):
        super().__init__(fraud_augmentation_factor, use_enhanced_augmentation)


def main():
    """Test the enhanced data preprocessing pipeline"""
    from config import DATASET_CONFIG
    
    # Test with enhanced features
    print("Testing Enhanced Data Preprocessing:")
    enhanced_preprocessor = EnhancedDataPreprocessor(
        fraud_augmentation_factor=8, 
        use_enhanced_augmentation=True
    )
    
    # Analyze dataset quality
    analysis = enhanced_preprocessor.analyze_dataset_quality(DATASET_CONFIG["train_dir"])
    print("Dataset Analysis:")
    for rec in analysis['recommendations']:
        print(f"- {rec}")
    
    # Create enhanced data loaders
    train_loader, test_loader = enhanced_preprocessor.create_data_loaders(
        DATASET_CONFIG["train_dir"],
        DATASET_CONFIG["test_dir"],
        batch_size=DATASET_CONFIG["batch_size"],
        use_enhanced=True
    )
    
    print(f"Enhanced training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels distribution: {Counter(labels.numpy())}")
        break
    
    # Test with standard features (backward compatibility)
    print("\nTesting Standard Data Preprocessing:")
    standard_preprocessor = DataPreprocessor(
        fraud_augmentation_factor=5, 
        use_enhanced_augmentation=False
    )
    
    train_loader_std, test_loader_std = standard_preprocessor.create_data_loaders(
        DATASET_CONFIG["train_dir"],
        DATASET_CONFIG["test_dir"],
        batch_size=DATASET_CONFIG["batch_size"],
        use_enhanced=False
    )
    
    print(f"Standard training batches: {len(train_loader_std)}")
    print(f"Standard test batches: {len(test_loader_std)}")


if __name__ == "__main__":
    main()