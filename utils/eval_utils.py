"""
Evaluation utilities for UFGVC with FAIM-Head.

This module provides comprehensive evaluation metrics, visualization tools,
and analysis functions for ultra-fine-grained visual classification tasks.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_curve, auc,
    top_k_accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from collections import defaultdict
import warnings

try:
    from ..models.faim_head import FAIMHead
except ImportError:
    from models.faim_head import FAIMHead


class UFGVCEvaluator:
    """Comprehensive evaluator for UFGVC tasks with FAIM-Head analysis.
    
    Args:
        model: Trained model with FAIM-Head
        device: Device for computation
        class_names: List of class names
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device,
        class_names: Optional[List[str]] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        
        # Find FAIM head for specialized analysis
        self.faim_head = self._find_faim_head()
        
    def _find_faim_head(self) -> Optional[FAIMHead]:
        """Find FAIM-Head in the model."""
        for module in self.model.modules():
            if isinstance(module, FAIMHead):
                return module
        return None
    
    def evaluate_comprehensive(
        self, 
        dataloader: torch.utils.data.DataLoader,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation including FAIM-specific metrics.
        
        Args:
            dataloader: Data loader for evaluation
            save_dir: Directory to save evaluation results
            
        Returns:
            Complete evaluation results
        """
        self.model.eval()
        
        # Storage for results
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_features = []
        all_distances = []
        all_logits = []
        
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                logits = self.model(images)
                loss = criterion(logits, targets)
                total_loss += loss.item()
                
                # Get probabilities and predictions
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Extract features if possible
                features = self._extract_features(images)
                if features is not None:
                    all_features.append(features.cpu())
                
                # Get FAIM distances if available
                if self.faim_head is not None and features is not None:
                    distances = self.faim_head.get_distances(features)
                    all_distances.append(distances.cpu())
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_probabilities.append(probabilities.cpu())
                all_targets.append(targets.cpu())
                all_logits.append(logits.cpu())
                
                if batch_idx % 50 == 0:
                    print(f"Evaluating batch {batch_idx}/{len(dataloader)}")
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions).numpy()
        all_probabilities = torch.cat(all_probabilities).numpy()
        all_targets = torch.cat(all_targets).numpy()
        all_logits = torch.cat(all_logits).numpy()
        
        if all_features:
            all_features = torch.cat(all_features).numpy()
        if all_distances:
            all_distances = torch.cat(all_distances).numpy()
        
        # Compute standard metrics
        standard_metrics = self._compute_standard_metrics(
            all_targets, all_predictions, all_probabilities, all_logits
        )
        
        # Compute FAIM-specific metrics
        faim_metrics = {}
        if all_distances:
            faim_metrics = self._compute_faim_metrics(all_distances, all_targets)
        
        # Feature analysis
        feature_analysis = {}
        if all_features:
            feature_analysis = self._analyze_features(all_features, all_targets)
        
        # Combine results
        results = {
            'standard_metrics': standard_metrics,
            'faim_metrics': faim_metrics,
            'feature_analysis': feature_analysis,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'targets': all_targets,
            'logits': all_logits,
            'average_loss': total_loss / len(dataloader)
        }
        
        if all_features:
            results['features'] = all_features
        if all_distances:
            results['distances'] = all_distances
        
        # Save results if directory provided
        if save_dir:
            self._save_evaluation_results(results, save_dir)
        
        return results
    
    def _extract_features(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract features from the model backbone."""
        try:
            # Try common feature extraction methods
            if hasattr(self.model, 'features'):
                features = self.model.features(images)
            elif hasattr(self.model, 'forward_features'):
                features = self.model.forward_features(images)
            else:
                # Generic approach: forward through all modules except head
                features = images
                for name, module in self.model.named_children():
                    if name != 'head':
                        features = module(features)
                    else:
                        break
            
            # Flatten if needed
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            
            return features
            
        except Exception as e:
            print(f"Warning: Could not extract features: {e}")
            return None
    
    def _compute_standard_metrics(
        self, 
        targets: np.ndarray, 
        predictions: np.ndarray,
        probabilities: np.ndarray,
        logits: np.ndarray
    ) -> Dict[str, Any]:
        """Compute standard classification metrics."""
        num_classes = len(np.unique(targets))
        
        # Basic accuracy metrics
        accuracy = np.mean(predictions == targets)
        
        # Top-k accuracies
        top_k_accs = {}
        for k in [1, 3, 5]:
            if k <= num_classes:
                top_k_accs[f'top_{k}_accuracy'] = top_k_accuracy_score(
                    targets, probabilities, k=k
                )
        
        # Classification report
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = classification_report(
                targets, predictions, 
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Per-class metrics
        per_class_metrics = {}
        for i in range(num_classes):
            class_mask = targets == i
            if np.sum(class_mask) > 0:
                class_predictions = predictions[class_mask]
                class_accuracy = np.mean(class_predictions == i)
                per_class_metrics[f'class_{i}_accuracy'] = class_accuracy
        
        return {
            'accuracy': accuracy,
            'top_k_accuracies': top_k_accs,
            'classification_report': report,
            'confusion_matrix': cm,
            'per_class_metrics': per_class_metrics,
            'num_classes': num_classes,
            'num_samples': len(targets)
        }
    
    def _compute_faim_metrics(
        self, 
        distances: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """Compute FAIM-specific metrics."""
        num_samples, num_classes = distances.shape
        
        # Get distances to correct classes
        correct_distances = distances[np.arange(num_samples), targets]
        
        # Get distances to incorrect classes
        mask = np.ones_like(distances, dtype=bool)
        mask[np.arange(num_samples), targets] = False
        incorrect_distances = distances[mask].reshape(num_samples, -1)
        
        # Compute margins (difference between closest incorrect and correct)
        closest_incorrect = np.min(incorrect_distances, axis=1)
        margins = closest_incorrect - correct_distances
        
        # Distance-based accuracy (closest prototype)
        distance_predictions = np.argmin(distances, axis=1)
        distance_accuracy = np.mean(distance_predictions == targets)
        
        # Margin statistics
        positive_margins = margins[margins > 0]
        negative_margins = margins[margins <= 0]
        
        # Class-wise distance analysis
        class_distance_stats = {}
        for class_idx in range(num_classes):
            class_mask = targets == class_idx
            if np.sum(class_mask) > 0:
                class_correct_distances = correct_distances[class_mask]
                class_distance_stats[f'class_{class_idx}'] = {
                    'mean_distance': np.mean(class_correct_distances),
                    'std_distance': np.std(class_correct_distances),
                    'min_distance': np.min(class_correct_distances),
                    'max_distance': np.max(class_correct_distances)
                }
        
        # FAIM parameter analysis
        faim_params = {}
        if self.faim_head is not None:
            faim_params = {
                'lambda': self.faim_head.lmbda.item(),
                'scale': self.faim_head.scale.item(),
                'sigma_eigenvalues': self.faim_head.get_sigma_eigenvalues().cpu().numpy(),
                'prototype_distances': self.faim_head.get_prototype_distances().cpu().numpy()
            }
        
        return {
            'distance_accuracy': distance_accuracy,
            'mean_correct_distance': np.mean(correct_distances),
            'mean_incorrect_distance': np.mean(incorrect_distances),
            'mean_margin': np.mean(margins),
            'margin_std': np.std(margins),
            'positive_margin_ratio': len(positive_margins) / len(margins),
            'negative_margin_count': len(negative_margins),
            'margin_distribution': {
                'positive_margins': positive_margins,
                'negative_margins': negative_margins
            },
            'class_distance_stats': class_distance_stats,
            'faim_parameters': faim_params,
            'raw_margins': margins,
            'correct_distances': correct_distances,
            'incorrect_distances': incorrect_distances.mean(axis=1)
        }
    
    def _analyze_features(
        self, 
        features: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze feature representations."""
        # Compute feature statistics
        feature_stats = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'feature_dim': features.shape[1]
        }
        
        # Compute class centroids
        num_classes = len(np.unique(targets))
        centroids = np.zeros((num_classes, features.shape[1]))
        
        for class_idx in range(num_classes):
            class_mask = targets == class_idx
            if np.sum(class_mask) > 0:
                centroids[class_idx] = np.mean(features[class_mask], axis=0)
        
        # Compute intra-class and inter-class distances
        intra_class_distances = []
        inter_class_distances = []
        
        for class_idx in range(num_classes):
            class_mask = targets == class_idx
            class_features = features[class_mask]
            
            if len(class_features) > 1:
                # Intra-class distances
                centroid = centroids[class_idx]
                distances_to_centroid = np.linalg.norm(class_features - centroid, axis=1)
                intra_class_distances.extend(distances_to_centroid)
            
            # Inter-class distances (to other centroids)
            for other_class_idx in range(num_classes):
                if other_class_idx != class_idx:
                    inter_distance = np.linalg.norm(centroids[class_idx] - centroids[other_class_idx])
                    inter_class_distances.append(inter_distance)
        
        # Silhouette-like analysis
        separability_score = 0.0
        if intra_class_distances and inter_class_distances:
            mean_intra = np.mean(intra_class_distances)
            mean_inter = np.mean(inter_class_distances)
            separability_score = (mean_inter - mean_intra) / max(mean_inter, mean_intra)
        
        return {
            'feature_stats': feature_stats,
            'centroids': centroids,
            'mean_intra_class_distance': np.mean(intra_class_distances) if intra_class_distances else 0.0,
            'mean_inter_class_distance': np.mean(inter_class_distances) if inter_class_distances else 0.0,
            'separability_score': separability_score,
            'intra_class_distances': intra_class_distances,
            'inter_class_distances': inter_class_distances
        }
    
    def _save_evaluation_results(self, results: Dict[str, Any], save_dir: str):
        """Save evaluation results to files."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results (excluding large arrays)
        main_results = {}
        for key, value in results.items():
            if key not in ['features', 'distances', 'predictions', 'probabilities', 
                          'targets', 'logits']:
                if isinstance(value, dict):
                    # Recursively handle nested dictionaries
                    main_results[key] = self._prepare_for_json(value)
                else:
                    main_results[key] = value
        
        with open(save_path / 'evaluation_results.json', 'w') as f:
            json.dump(main_results, f, indent=2, default=str)
        
        # Save arrays separately
        np.save(save_path / 'predictions.npy', results['predictions'])
        np.save(save_path / 'targets.npy', results['targets'])
        np.save(save_path / 'probabilities.npy', results['probabilities'])
        
        if 'features' in results:
            np.save(save_path / 'features.npy', results['features'])
        if 'distances' in results:
            np.save(save_path / 'distances.npy', results['distances'])
        
        print(f"Evaluation results saved to {save_path}")
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            if obj.size < 1000:  # Only save small arrays
                return obj.tolist()
            else:
                return f"<array shape={obj.shape} dtype={obj.dtype}>"
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def plot_confusion_matrix(
        self, 
        cm: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """Plot confusion matrix."""
        plt.figure(figsize=figsize)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True if cm.shape[0] <= 20 else False,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names if self.class_names and len(self.class_names) <= 20 else False,
            yticklabels=self.class_names if self.class_names and len(self.class_names) <= 20 else False
        )
        
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_margin_distribution(
        self, 
        margins: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """Plot distribution of FAIM margins."""
        plt.figure(figsize=figsize)
        
        # Plot histogram
        plt.hist(margins, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero margin')
        plt.axvline(x=np.mean(margins), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(margins):.3f}')
        
        plt.xlabel('Margin (Closest Incorrect - Correct Distance)')
        plt.ylabel('Frequency')
        plt.title('Distribution of FAIM Margins')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {np.mean(margins):.3f}\n'
        stats_text += f'Std: {np.std(margins):.3f}\n'
        stats_text += f'Positive margins: {np.sum(margins > 0)}/{len(margins)} ({100*np.sum(margins > 0)/len(margins):.1f}%)'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_tsne(
        self, 
        features: np.ndarray,
        targets: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        n_samples: int = 5000
    ):
        """Plot t-SNE visualization of features."""
        # Sample data if too large
        if len(features) > n_samples:
            indices = np.random.choice(len(features), n_samples, replace=False)
            features = features[indices]
            targets = targets[indices]
        
        # Compute t-SNE
        print("Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)
        
        # Plot
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            features_2d[:, 0], features_2d[:, 1], 
            c=targets, cmap='tab20', alpha=0.6, s=10
        )
        
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Features')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_distance_heatmap(
        self, 
        prototype_distances: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Plot heatmap of distances between class prototypes."""
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            prototype_distances,
            annot=True if prototype_distances.shape[0] <= 20 else False,
            fmt='.3f',
            cmap='viridis',
            xticklabels=self.class_names if self.class_names and len(self.class_names) <= 20 else False,
            yticklabels=self.class_names if self.class_names and len(self.class_names) <= 20 else False
        )
        
        plt.title('Distances Between Class Prototypes')
        plt.xlabel('Class')
        plt.ylabel('Class')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(
        self, 
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """Generate a comprehensive evaluation report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("UFGVC EVALUATION REPORT WITH FAIM-HEAD")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Standard metrics
        std_metrics = results['standard_metrics']
        report_lines.append("STANDARD CLASSIFICATION METRICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Overall Accuracy: {std_metrics['accuracy']:.4f}")
        report_lines.append(f"Number of Classes: {std_metrics['num_classes']}")
        report_lines.append(f"Number of Samples: {std_metrics['num_samples']}")
        report_lines.append("")
        
        # Top-k accuracies
        if 'top_k_accuracies' in std_metrics:
            report_lines.append("Top-k Accuracies:")
            for k, acc in std_metrics['top_k_accuracies'].items():
                report_lines.append(f"  {k}: {acc:.4f}")
            report_lines.append("")
        
        # FAIM-specific metrics
        if 'faim_metrics' in results and results['faim_metrics']:
            faim_metrics = results['faim_metrics']
            report_lines.append("FAIM-SPECIFIC METRICS")
            report_lines.append("-" * 40)
            report_lines.append(f"Distance-based Accuracy: {faim_metrics['distance_accuracy']:.4f}")
            report_lines.append(f"Mean Correct Distance: {faim_metrics['mean_correct_distance']:.4f}")
            report_lines.append(f"Mean Incorrect Distance: {faim_metrics['mean_incorrect_distance']:.4f}")
            report_lines.append(f"Mean Margin: {faim_metrics['mean_margin']:.4f}")
            report_lines.append(f"Margin Std: {faim_metrics['margin_std']:.4f}")
            report_lines.append(f"Positive Margin Ratio: {faim_metrics['positive_margin_ratio']:.4f}")
            report_lines.append("")
            
            # FAIM parameters
            if 'faim_parameters' in faim_metrics:
                params = faim_metrics['faim_parameters']
                report_lines.append("FAIM Parameters:")
                report_lines.append(f"  Lambda (λ): {params.get('lambda', 'N/A')}")
                report_lines.append(f"  Scale (γ): {params.get('scale', 'N/A')}")
                if 'sigma_eigenvalues' in params:
                    eigenvals = params['sigma_eigenvalues']
                    report_lines.append(f"  Σ Eigenvalue range: [{eigenvals.min():.4f}, {eigenvals.max():.4f}]")
                report_lines.append("")
        
        # Feature analysis
        if 'feature_analysis' in results and results['feature_analysis']:
            feat_analysis = results['feature_analysis']
            report_lines.append("FEATURE ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Feature Dimension: {feat_analysis['feature_stats']['feature_dim']}")
            report_lines.append(f"Mean Intra-class Distance: {feat_analysis['mean_intra_class_distance']:.4f}")
            report_lines.append(f"Mean Inter-class Distance: {feat_analysis['mean_inter_class_distance']:.4f}")
            report_lines.append(f"Separability Score: {feat_analysis['separability_score']:.4f}")
            report_lines.append("")
        
        # Classification report summary
        if 'classification_report' in std_metrics:
            report = std_metrics['classification_report']
            if 'macro avg' in report:
                macro_avg = report['macro avg']
                report_lines.append("MACRO-AVERAGED METRICS")
                report_lines.append("-" * 40)
                report_lines.append(f"Precision: {macro_avg['precision']:.4f}")
                report_lines.append(f"Recall: {macro_avg['recall']:.4f}")
                report_lines.append(f"F1-Score: {macro_avg['f1-score']:.4f}")
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Evaluation report saved to {save_path}")
        
        return report_text


def evaluate_model_comprehensive(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function for comprehensive model evaluation.
    
    Args:
        model: Trained model with FAIM-Head
        dataloader: Data loader for evaluation
        device: Device for computation
        class_names: List of class names
        save_dir: Directory to save results
        
    Returns:
        Complete evaluation results
    """
    evaluator = UFGVCEvaluator(model, device, class_names)
    results = evaluator.evaluate_comprehensive(dataloader, save_dir)
    
    # Generate plots if save directory provided
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        cm = results['standard_metrics']['confusion_matrix']
        evaluator.plot_confusion_matrix(cm, str(save_path / 'confusion_matrix.png'))
        
        # FAIM margins
        if 'faim_metrics' in results and 'raw_margins' in results['faim_metrics']:
            margins = results['faim_metrics']['raw_margins']
            evaluator.plot_margin_distribution(margins, str(save_path / 'margin_distribution.png'))
        
        # Feature visualization
        if 'features' in results:
            features = results['features']
            targets = results['targets']
            evaluator.plot_feature_tsne(features, targets, str(save_path / 'feature_tsne.png'))
        
        # Prototype distances
        if ('faim_metrics' in results and 'faim_parameters' in results['faim_metrics'] 
            and 'prototype_distances' in results['faim_metrics']['faim_parameters']):
            proto_distances = results['faim_metrics']['faim_parameters']['prototype_distances']
            evaluator.plot_distance_heatmap(proto_distances, str(save_path / 'prototype_distances.png'))
        
        # Generate report
        report = evaluator.generate_evaluation_report(results, str(save_path / 'evaluation_report.txt'))
        print("\nEVALUATION REPORT")
        print(report)
    
    return results


if __name__ == "__main__":
    print("Evaluation utilities ready for use!")
    print("Use evaluate_model_comprehensive() for complete evaluation with FAIM-Head analysis.")
