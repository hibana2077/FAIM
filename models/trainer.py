"""
Training utilities and pipeline for UFGVC with FAIM-Head.

This module provides comprehensive training functionality including:
- Multi-phase training (warm-up, tune-metric, fine-tune)
- Learning rate scheduling
- Metrics computation and tracking
- Model evaluation utilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .faim_head import FAIMHead, FAIMContrastiveLoss
except ImportError:
    from faim_head import FAIMHead, FAIMContrastiveLoss


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FAIMTrainer:
    """Comprehensive trainer for UFGVC with FAIM-Head.
    
    Implements the three-phase training strategy:
    1. Warm-up: Freeze metric parameters, train only prototypes and backbone
    2. Tune-metric: Unfreeze β and λ, gradually unfreeze Σ
    3. Fine-tune: Train all parameters together
    
    Args:
        model: Model with FAIM-Head
        device: Device to train on
        log_dir: Directory for logging and checkpoints
        use_contrastive: Whether to use contrastive loss alongside CrossEntropy
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        log_dir: str = "./logs",
        use_contrastive: bool = False,
        contrastive_weight: float = 0.1
    ):
        self.model = model.to(device)
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss functions
        self.criterion_ce = nn.CrossEntropyLoss()
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        if use_contrastive:
            self.criterion_contrastive = FAIMContrastiveLoss()
        
        # Training state
        self.current_phase = "warmup"
        self.epoch = 0
        self.best_acc = 0.0
        self.history = defaultdict(list)
        
        # Setup logging
        self._setup_logging()
        
        # Find FAIM head
        self.faim_head = self._find_faim_head()
        if self.faim_head is None:
            raise ValueError("Model does not contain a FAIM-Head")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.log_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _find_faim_head(self) -> Optional[FAIMHead]:
        """Find FAIM-Head in the model."""
        for module in self.model.modules():
            if isinstance(module, FAIMHead):
                return module
        return None
    
    def train_phase(
        self,
        phase: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine",
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        """Train a specific phase.
        
        Args:
            phase: Training phase ('warmup', 'tune_metric', 'finetune')
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs for this phase
            lr: Learning rate
            weight_decay: Weight decay
            scheduler_type: Type of learning rate scheduler
            save_best: Whether to save best model
            
        Returns:
            Training history for this phase
        """
        self.logger.info(f"Starting {phase} phase for {epochs} epochs")
        self.current_phase = phase
        
        # Configure parameters based on phase
        self._configure_phase(phase)
        
        # Setup optimizer
        optimizer = self._create_optimizer(lr, weight_decay)
        
        # Setup scheduler
        scheduler = self._create_scheduler(optimizer, epochs, scheduler_type)
        
        # Training loop
        phase_history = defaultdict(list)
        
        for epoch in range(epochs):
            self.epoch += 1
            
            # Train epoch
            train_metrics = self._train_epoch(train_loader, optimizer)
            
            # Validate epoch
            val_metrics = self._validate_epoch(val_loader)
            
            # Update scheduler
            if scheduler is not None:
                if scheduler_type == "plateau":
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            # Log metrics
            self._log_epoch_metrics(train_metrics, val_metrics, optimizer.param_groups[0]['lr'])
            
            # Save history
            for key, value in train_metrics.items():
                phase_history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                phase_history[f'val_{key}'].append(value)
            phase_history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Save best model
            if save_best and val_metrics['acc1'] > self.best_acc:
                self.best_acc = val_metrics['acc1']
                self._save_checkpoint('best', val_metrics['acc1'])
                self.logger.info(f"New best accuracy: {self.best_acc:.3f}%")
        
        # Save final checkpoint for this phase
        self._save_checkpoint(f'{phase}_final', val_metrics['acc1'])
        
        return dict(phase_history)
    
    def _configure_phase(self, phase: str):
        """Configure model parameters for specific training phase."""
        if phase == "warmup":
            # Freeze metric parameters, allow prototypes and backbone
            self.faim_head.freeze_metric_params()
            self.faim_head.mu.requires_grad_(True)
            
        elif phase == "tune_metric":
            # Unfreeze β and λ, gradually unfreeze Σ
            self.faim_head.unfreeze_metric_params()
            
            # Start with lower learning rate for newly unfrozen parameters
            for param in [self.faim_head.beta, self.faim_head.lmbda]:
                if hasattr(param, '_lr_mult'):
                    delattr(param, '_lr_mult')
                param._lr_mult = 0.1  # Lower learning rate multiplier
            
        elif phase == "finetune":
            # All parameters trainable
            self.faim_head.unfreeze_metric_params()
            
            # Remove learning rate multipliers
            for param in self.faim_head.parameters():
                if hasattr(param, '_lr_mult'):
                    delattr(param, '_lr_mult')
        
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    def _create_optimizer(self, lr: float, weight_decay: float) -> optim.Optimizer:
        """Create optimizer with parameter-specific learning rates."""
        param_groups = []
        
        # Backbone parameters
        backbone_params = []
        for name, param in self.model.named_parameters():
            if not name.startswith('head.') and param.requires_grad:
                backbone_params.append(param)
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': lr,
                'weight_decay': weight_decay
            })
        
        # FAIM head parameters with potentially different learning rates
        head_params = []
        for param in self.faim_head.parameters():
            if param.requires_grad:
                lr_mult = getattr(param, '_lr_mult', 1.0)
                param_groups.append({
                    'params': [param],
                    'lr': lr * lr_mult,
                    'weight_decay': weight_decay if param.dim() > 1 else 0.0
                })
        
        return optim.AdamW(param_groups)
    
    def _create_scheduler(
        self, 
        optimizer: optim.Optimizer, 
        epochs: int, 
        scheduler_type: str
    ) -> Optional[object]:
        """Create learning rate scheduler."""
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
        elif scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
        elif scheduler_type == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        losses = AverageMeter()
        ce_losses = AverageMeter()
        if self.use_contrastive:
            cont_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        start_time = time.time()
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(images)
            
            # Compute losses
            ce_loss = self.criterion_ce(logits, targets)
            total_loss = ce_loss
            
            if self.use_contrastive and isinstance(self.model.head, FAIMHead):
                # Get distances for contrastive loss
                distances = self.faim_head.get_distances(
                    self.model.features(images) if hasattr(self.model, 'features') 
                    else self._get_features(images)
                )
                cont_loss = self.criterion_contrastive(distances, targets)
                total_loss += self.contrastive_weight * cont_loss
                cont_losses.update(cont_loss.item(), images.size(0))
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            acc1, acc5 = self._accuracy(logits, targets, topk=(1, 5))
            losses.update(total_loss.item(), images.size(0))
            ce_losses.update(ce_loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            
            # Log batch progress
            if batch_idx % 100 == 0:
                self.logger.info(
                    f'Epoch {self.epoch} [{batch_idx}/{len(train_loader)}] '
                    f'Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f}%'
                )
        
        epoch_time = time.time() - start_time
        
        metrics = {
            'loss': losses.avg,
            'ce_loss': ce_losses.avg,
            'acc1': top1.avg,
            'acc5': top5.avg,
            'time': epoch_time
        }
        
        if self.use_contrastive:
            metrics['cont_loss'] = cont_losses.avg
            
        return metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                logits = self.model(images)
                loss = self.criterion_ce(logits, targets)
                
                # Update metrics
                acc1, acc5 = self._accuracy(logits, targets, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
        
        return {
            'loss': losses.avg,
            'acc1': top1.avg,
            'acc5': top5.avg
        }
    
    def _get_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone (before classification head)."""
        # For timm models, we need to be more careful about feature extraction
        if hasattr(self.model, 'forward_features'):
            # Use the forward_features method if available (common in timm models)
            features = self.model.forward_features(images)
        else:
            # Fallback: manually iterate through layers
            features = images
            for name, module in self.model.named_children():
                if name not in ['head', 'classifier', 'fc']:
                    features = module(features)
                else:
                    break
        
        # Handle different feature shapes
        if features.dim() > 2:
            # For ViT models, features might be [B, seq_len, dim] - take the CLS token or global average
            if features.dim() == 3:
                # Assume this is [batch, sequence, features] from ViT
                # Take the first token (CLS token) or average pool
                if hasattr(self.model, 'global_pool') and self.model.global_pool == 'avg':
                    features = features.mean(dim=1)
                else:
                    features = features[:, 0]  # CLS token
            else:
                # Flatten other cases
                features = features.view(features.size(0), -1)
        
        return features
    
    @staticmethod
    def _accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
        """Compute accuracy for specified k values."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            
            return res
    
    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict, lr: float):
        """Log metrics for current epoch."""
        self.logger.info(
            f'Epoch {self.epoch} ({self.current_phase}) - '
            f'LR: {lr:.2e} | '
            f'Train Loss: {train_metrics["loss"]:.4f} Acc@1: {train_metrics["acc1"]:.2f}% | '
            f'Val Loss: {val_metrics["loss"]:.4f} Acc@1: {val_metrics["acc1"]:.2f}%'
        )
        
        # Log FAIM-specific metrics
        if isinstance(self.faim_head, FAIMHead):
            self.logger.info(
                f'FAIM params - λ: {self.faim_head.lmbda.item():.4f} '
                f'scale: {self.faim_head.scale.item():.2f}'
            )
    
    def _save_checkpoint(self, name: str, acc: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'phase': self.current_phase,
            'model_state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'acc': acc,
            'history': dict(self.history)
        }
        
        checkpoint_path = self.log_dir / f'{name}_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'Saved checkpoint: {checkpoint_path}')
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        self.current_phase = checkpoint['phase']
        self.best_acc = checkpoint['best_acc']
        self.history = defaultdict(list, checkpoint.get('history', {}))
        
        self.logger.info(f'Loaded checkpoint from epoch {self.epoch} with acc {checkpoint["acc"]:.3f}%')
        return checkpoint
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_distances = []
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                logits = self.model(images)
                loss = self.criterion_ce(logits, targets)
                
                # Get predictions
                _, predictions = logits.topk(1, 1, True, True)
                predictions = predictions.squeeze(1)
                
                # Get distances if FAIM head
                if isinstance(self.faim_head, FAIMHead):
                    features = self._get_features(images)
                    distances = self.faim_head.get_distances(features)
                    all_distances.append(distances.cpu())
                
                # Update metrics
                acc1, acc5 = self._accuracy(logits, targets, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))
                
                # Store for detailed analysis
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        if all_distances:
            all_distances = torch.cat(all_distances)
        
        # Compute detailed metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        report = classification_report(
            all_targets.numpy(), 
            all_predictions.numpy(), 
            output_dict=True,
            zero_division=0
        )
        
        results = {
            'loss': losses.avg,
            'acc1': top1.avg,
            'acc5': top5.avg,
            'classification_report': report,
            'predictions': all_predictions.numpy(),
            'targets': all_targets.numpy(),
        }
        
        if all_distances:
            results['distances'] = all_distances.numpy()
            
            # FAIM-specific analysis
            results['faim_analysis'] = self._analyze_faim_metrics(all_distances, all_targets)
        
        return results
    
    def _analyze_faim_metrics(self, distances: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Analyze FAIM-specific metrics."""
        num_classes = distances.shape[1]
        
        # Compute mean distances to correct vs incorrect classes
        correct_distances = distances.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Create mask for incorrect classes
        incorrect_mask = torch.ones_like(distances).bool()
        incorrect_mask.scatter_(1, targets.unsqueeze(1), False)
        incorrect_distances = distances.masked_select(incorrect_mask)
        
        # Compute margins (difference between closest incorrect and correct)
        distances_masked = distances.clone()
        distances_masked.scatter_(1, targets.unsqueeze(1), float('inf'))
        closest_incorrect = distances_masked.min(dim=1)[0]
        margins = closest_incorrect - correct_distances
        
        return {
            'mean_correct_distance': correct_distances.mean().item(),
            'mean_incorrect_distance': incorrect_distances.mean().item(),
            'mean_margin': margins.mean().item(),
            'margin_std': margins.std().item(),
            'negative_margins': (margins < 0).sum().item(),
            'eigenvalues': self.faim_head.get_sigma_eigenvalues().cpu().numpy(),
            'lambda': self.faim_head.lmbda.item(),
            'scale': self.faim_head.scale.item(),
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        if not self.history:
            self.logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        if 'train_loss' in self.history:
            axes[0, 0].plot(self.history['train_loss'], label='Train')
            axes[0, 0].plot(self.history['val_loss'], label='Val')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].legend()
        
        # Accuracy plot
        if 'train_acc1' in self.history:
            axes[0, 1].plot(self.history['train_acc1'], label='Train')
            axes[0, 1].plot(self.history['val_acc1'], label='Val')
            axes[0, 1].set_title('Top-1 Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].legend()
        
        # Learning rate
        if 'lr' in self.history:
            axes[1, 0].plot(self.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_yscale('log')
        
        # FAIM parameters (if available)
        axes[1, 1].text(0.1, 0.5, f'Final λ: {self.faim_head.lmbda.item():.4f}\n'
                                  f'Final scale: {self.faim_head.scale.item():.2f}',
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('FAIM Parameters')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()


def train_full_pipeline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    log_dir: str = "./logs"
) -> Dict[str, Any]:
    """Complete training pipeline with all three phases.
    
    Args:
        model: Model with FAIM-Head
        train_loader: Training data loader
        val_loader: Validation data loader  
        test_loader: Test data loader
        device: Device to train on
        config: Training configuration
        log_dir: Directory for logging
        
    Returns:
        Complete training results and final evaluation
    """
    trainer = FAIMTrainer(model, device, log_dir, **config.get('trainer', {}))
    
    # Phase 1: Warm-up
    warmup_config = config.get('warmup', {})
    warmup_history = trainer.train_phase(
        'warmup',
        train_loader,
        val_loader,
        **warmup_config
    )
    
    # Phase 2: Tune-metric
    tune_config = config.get('tune_metric', {})
    tune_history = trainer.train_phase(
        'tune_metric',
        train_loader,
        val_loader,
        **tune_config
    )
    
    # Phase 3: Fine-tune
    finetune_config = config.get('finetune', {})
    finetune_history = trainer.train_phase(
        'finetune',
        train_loader,
        val_loader,
        **finetune_config
    )
    
    # Final evaluation
    final_results = trainer.evaluate_model(test_loader)
    
    # Plot training history
    trainer.plot_training_history(str(Path(log_dir) / 'training_history.png'))
    
    # Save final results
    results = {
        'warmup_history': warmup_history,
        'tune_history': tune_history, 
        'finetune_history': finetune_history,
        'final_evaluation': final_results,
        'best_accuracy': trainer.best_acc
    }
    
    results_path = Path(log_dir) / 'final_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: v.tolist() if hasattr(v, 'tolist') else v 
                                   for k, v in value.items() if not isinstance(v, (torch.Tensor, np.ndarray))}
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2)
    
    trainer.logger.info(f"Training completed! Best accuracy: {trainer.best_acc:.3f}%")
    trainer.logger.info(f"Results saved to {results_path}")
    
    return results
