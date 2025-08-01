#!/usr/bin/env python3
"""
Main training script for UFGVC with FAIM-Head.

This script provides a complete pipeline for training ultra-fine-grained visual 
classification models using the Finsler-α Information Manifold (FAIM) classification head.

Features:
- Three-phase training strategy (warm-up, tune-metric, fine-tune)
- Support for multiple UFGVC datasets
- Comprehensive evaluation with FAIM-specific metrics
- Experiment tracking and result visualization
- Configurable hyperparameters and model architectures

Usage:
    python train_faim.py --dataset cotton80 --model vit_base_patch16_224 --config configs/default.yaml
    python train_faim.py --dataset soybean --quick fast_training
    python train_faim.py --help
"""

import argparse
import torch
import torch.nn as nn
import random
import numpy as np
import warnings
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import create_faim_model, train_full_pipeline
from config import get_config, get_quick_config, validate_config, print_config
from utils import create_data_module, evaluate_model_comprehensive
from dataset import UFGVCDataset


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = 'auto') -> torch.device:
    """Get the appropriate device for training."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")
    else:
        device = torch.device(device_str)
        print(f"Using specified device: {device}")
    
    return device


def create_experiment_name(config: dict) -> str:
    """Create a unique experiment name based on configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = config['data']['dataset_name']
    model_name = config['model']['name'].replace('/', '_')
    lambda_val = config['model']['faim_head']['lambda_init']
    
    return f"{dataset_name}_{model_name}_lambda{lambda_val}_{timestamp}"


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train UFGVC models with FAIM-Head",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and model arguments
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='cotton80',
        choices=['cotton80', 'soybean', 'soy_ageing_r1', 'soy_ageing_r3', 
                'soy_ageing_r4', 'soy_ageing_r5', 'soy_ageing_r6'],
        help='Dataset to use for training'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='vit_base_patch16_224',
        help='Model architecture (timm model name)'
    )
    
    # Configuration arguments
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--quick', 
        type=str, 
        default=None,
        choices=['small_experiment', 'fast_training', 'high_quality'],
        help='Use a quick start configuration template'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--lr', 
        type=float, 
        default=None,
        help='Learning rate for all phases (overrides config)'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        nargs=3,
        default=None,
        metavar=('WARMUP', 'TUNE', 'FINETUNE'),
        help='Number of epochs for each phase (warmup, tune-metric, finetune)'
    )
    
    parser.add_argument(
        '--lambda_init', 
        type=float, 
        default=None,
        help='Initial lambda parameter for FAIM-Head'
    )
    
    parser.add_argument(
        '--scale_init', 
        type=float, 
        default=None,
        help='Initial scale parameter for FAIM-Head'
    )
    
    parser.add_argument(
        '--full_sigma', 
        action='store_true',
        help='Use full Sigma matrix instead of diagonal'
    )
    
    # System arguments
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        help='Device to use (auto, cpu, cuda, cuda:0, etc.)'
    )
    
    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=4,
        help='Number of data loading workers'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Output arguments
    parser.add_argument(
        '--log_dir', 
        type=str, 
        default='./logs',
        help='Directory for logs and checkpoints'
    )
    
    parser.add_argument(
        '--experiment_name', 
        type=str, 
        default=None,
        help='Experiment name (auto-generated if not provided)'
    )
    
    parser.add_argument(
        '--no_eval', 
        action='store_true',
        help='Skip final evaluation'
    )
    
    parser.add_argument(
        '--save_model', 
        action='store_true',
        help='Save final model state dict'
    )
    
    # Debug arguments
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode (smaller dataset, fewer epochs)'
    )
    
    parser.add_argument(
        '--dry_run', 
        action='store_true',
        help='Load config and data but do not train'
    )
    
    parser.add_argument(
        '--list_datasets', 
        action='store_true',
        help='List available datasets and exit'
    )
    
    args = parser.parse_args()
    
    # List datasets if requested
    if args.list_datasets:
        print("Available UFGVC datasets:")
        for name, desc in UFGVCDataset.list_available_datasets().items():
            print(f"  {name}: {desc}")
        return
    
    # Set random seed
    set_seed(args.seed)
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    print("=" * 80)
    print("UFGVC TRAINING WITH FAIM-HEAD")
    print("=" * 80)
    
    # Get configuration
    print("\\n📋 Loading configuration...")
    
    if args.quick:
        config = get_quick_config(
            args.quick, 
            dataset_name=args.dataset,
            model_name=args.model,
            config_file=args.config
        )
        print(f"Using quick config template: {args.quick}")
    else:
        config = get_config(
            dataset_name=args.dataset,
            model_name=args.model,
            config_file=args.config
        )
    
    # Apply command line overrides
    overrides = {}
    
    if args.batch_size:
        overrides['data'] = {'batch_size': args.batch_size}
    
    if args.lr:
        overrides['warmup'] = {'lr': args.lr}
        overrides['tune_metric'] = {'lr': args.lr * 0.5}
        overrides['finetune'] = {'lr': args.lr * 0.1}
    
    if args.epochs:
        overrides['warmup'] = overrides.get('warmup', {})
        overrides['tune_metric'] = overrides.get('tune_metric', {})
        overrides['finetune'] = overrides.get('finetune', {})
        overrides['warmup']['epochs'] = args.epochs[0]
        overrides['tune_metric']['epochs'] = args.epochs[1]
        overrides['finetune']['epochs'] = args.epochs[2]
    
    if args.lambda_init:
        overrides['model'] = {'faim_head': {'lambda_init': args.lambda_init}}
    
    if args.scale_init:
        overrides['model'] = overrides.get('model', {})
        overrides['model']['faim_head'] = overrides['model'].get('faim_head', {})
        overrides['model']['faim_head']['scale_init'] = args.scale_init
    
    if args.full_sigma:
        overrides['model'] = overrides.get('model', {})
        overrides['model']['faim_head'] = overrides['model'].get('faim_head', {})
        overrides['model']['faim_head']['full_sigma'] = True
    
    if args.num_workers:
        overrides['data'] = overrides.get('data', {})
        overrides['data']['num_workers'] = args.num_workers
    
    # Apply overrides
    if overrides:
        from config.settings import deep_merge_dict
        config = deep_merge_dict(config, overrides)
    
    # Debug mode adjustments
    if args.debug:
        print("🐛 Debug mode enabled")
        config['warmup']['epochs'] = 2
        config['tune_metric']['epochs'] = 3
        config['finetune']['epochs'] = 2
        config['data']['batch_size'] = min(config['data']['batch_size'], 8)
        config['data']['num_workers'] = 0
    
    # Validate configuration
    try:
        validate_config(config)
        print("✅ Configuration validated successfully")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return 1
    
    # Print configuration
    print_config(config, "Final Configuration")
    
    # Create experiment directory
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = create_experiment_name(config)
    
    log_dir = Path(args.log_dir) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Experiment directory: {log_dir}")
    
    # Save configuration
    config_path = log_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # Get device
    device = get_device(args.device)
    
    if args.dry_run:
        print("🏃 Dry run mode - exiting before training")
        return 0
    
    try:
        # Create data loaders
        print("\\n📊 Creating data loaders...")
        dataloaders = create_data_module(config)
        
        if not dataloaders:
            print("❌ No data loaders created")
            return 1
        
        print(f"Created data loaders: {list(dataloaders.keys())}")
        
        # Validate required splits
        required_splits = ['train']
        available_splits = list(dataloaders.keys())
        missing_splits = [split for split in required_splits if split not in available_splits]
        
        if missing_splits:
            print(f"❌ Missing required splits: {missing_splits}")
            return 1
        
        # Use validation split if available, otherwise use train for validation
        val_loader = dataloaders.get('val', dataloaders['train'])
        test_loader = dataloaders.get('test', val_loader)
        
        # Get dataset info
        sample_batch = next(iter(dataloaders['train']))
        sample_images, sample_labels = sample_batch
        
        # Infer number of classes
        if 'train' in dataloaders:
            train_dataset = dataloaders['train'].dataset
            num_classes = len(train_dataset.classes)
            class_names = train_dataset.classes
        else:
            # Fallback: try to infer from labels
            all_labels = []
            for _, labels in dataloaders[available_splits[0]]:
                all_labels.extend(labels.tolist())
            num_classes = len(set(all_labels))
            class_names = [f"Class_{i}" for i in range(num_classes)]
        
        print(f"📈 Dataset: {config['data']['dataset_name']}")
        print(f"📈 Number of classes: {num_classes}")
        print(f"📈 Image shape: {sample_images.shape[1:]}")
        
        # Create model
        print("\\n🏗️ Creating model...")
        model = create_faim_model(
            model_name=config['model']['name'],
            num_classes=num_classes,
            pretrained=config['model']['pretrained'],
            **config['model']['faim_head']
        )
        
        print(f"Model: {config['model']['name']}")
        print(f"FAIM-Head parameters: {config['model']['faim_head']}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        faim_params = sum(p.numel() for p in model.head.parameters())
        
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        print(f"📊 FAIM-Head parameters: {faim_params:,}")
        
        # Test forward pass
        model.to(device)
        with torch.no_grad():
            test_input = sample_images[:2].to(device)
            test_output = model(test_input)
            print(f"✅ Model forward pass successful: {test_input.shape} -> {test_output.shape}")
        
        # Training
        print("\\n🚀 Starting training...")
        training_results = train_full_pipeline(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            config=config,
            log_dir=str(log_dir)
        )
        
        print(f"\\n🎉 Training completed!")
        print(f"Best validation accuracy: {training_results['best_accuracy']:.3f}%")
        
        # Final evaluation
        if not args.no_eval and test_loader:
            print("\\n📊 Running final evaluation...")
            
            eval_results = evaluate_model_comprehensive(
                model=model,
                dataloader=test_loader,
                device=device,
                class_names=class_names,
                save_dir=str(log_dir / 'final_evaluation')
            )
            
            final_acc = eval_results['standard_metrics']['accuracy']
            print(f"Final test accuracy: {final_acc:.3f}%")
            
            # Save evaluation results
            training_results['final_evaluation'] = eval_results
        
        # Save model if requested
        if args.save_model:
            model_path = log_dir / 'final_model.pth'
            torch.save(model.state_dict(), model_path)
            print(f"💾 Model saved to {model_path}")
        
        # Save complete results
        results_path = log_dir / 'complete_results.json'
        with open(results_path, 'w') as f:
            # Convert results for JSON serialization
            json_results = {}
            for key, value in training_results.items():
                if key != 'final_evaluation':  # Skip large evaluation results
                    if isinstance(value, dict):
                        json_results[key] = {k: v for k, v in value.items() 
                                           if not isinstance(v, (torch.Tensor, np.ndarray))}
                    else:
                        json_results[key] = value
            
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\\n📋 Complete results saved to {log_dir}")
        print("\\n✅ Experiment completed successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\\n⏹️ Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
