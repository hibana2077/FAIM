#!/usr/bin/env python3
"""
Simple example script demonstrating UFGVC training with FAIM-Head.

This script shows the basic usage of the FAIM pipeline with minimal configuration.
Perfect for getting started quickly or testing the implementation.
"""

import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import create_faim_model, FAIMTrainer
from config import get_quick_config
from utils import create_data_module
from dataset import UFGVCDataset


def simple_example():
    """Run a simple training example with FAIM-Head."""
    print("🚀 FAIM-Head Simple Example")
    print("=" * 50)
    
    # 1. Get a quick configuration
    print("1. Loading configuration...")
    config = get_quick_config(
        'small_experiment',  # Small experiment for quick testing
        dataset_name='cotton80',
        model_name='vit_base_patch16_224'
    )
    
    # Override for even smaller experiment
    config['warmup']['epochs'] = 2
    config['tune_metric']['epochs'] = 3
    config['finetune']['epochs'] = 2
    config['data']['batch_size'] = 16
    
    print(f"Dataset: {config['data']['dataset_name']}")
    print(f"Model: {config['model']['name']}")
    print(f"FAIM λ: {config['model']['faim_head']['lambda_init']}")
    
    # 2. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 3. Create data loaders
    print("\\n2. Creating data loaders...")
    try:
        dataloaders = create_data_module(config)
        
        if 'train' not in dataloaders:
            print("❌ No training data available")
            return
        
        train_loader = dataloaders['train']
        val_loader = dataloaders.get('val', train_loader)
        test_loader = dataloaders.get('test', val_loader)
        
        # Get dataset info
        train_dataset = train_loader.dataset
        num_classes = len(train_dataset.classes)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Number of classes: {num_classes}")
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        print("This might be due to missing dataset. The dataset will be downloaded automatically.")
        return
    
    # 4. Create model
    print("\\n3. Creating FAIM model...")
    model = create_faim_model(
        model_name=config['model']['name'],
        num_classes=num_classes,
        pretrained=config['model']['pretrained'],
        **config['model']['faim_head']
    )
    
    # Count parameters
    faim_params = sum(p.numel() for p in model.head.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"FAIM-Head parameters: {faim_params:,}")
    print(f"Total model parameters: {total_params:,}")
    
    # 5. Test forward pass
    print("\\n4. Testing forward pass...")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_images, sample_labels = sample_batch
        sample_images = sample_images[:4].to(device)  # Test with 4 samples
        
        # Forward pass
        logits = model(sample_images)
        distances = model.head.get_distances(model.forward_features(sample_images) 
                                           if hasattr(model, 'forward_features') 
                                           else sample_images)
        
        print(f"Input shape: {sample_images.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Distance shape: {distances.shape}")
        print(f"Logit range: [{logits.min():.3f}, {logits.max():.3f}]")
        print(f"Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
    
    # 6. Quick training demonstration
    print("\\n5. Running quick training demo...")
    trainer = FAIMTrainer(model, device, log_dir="./logs/simple_example")
    
    try:
        # Just run warm-up phase for demonstration
        warmup_results = trainer.train_phase(
            phase='warmup',
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,  # Very short for demo
            lr=1e-3
        )
        
        print("✅ Warm-up phase completed!")
        print(f"Final train accuracy: {warmup_results['train_acc1'][-1]:.2f}%")
        print(f"Final val accuracy: {warmup_results['val_acc1'][-1]:.2f}%")
        
        # Show FAIM parameters
        print(f"\\nFAIM parameters after training:")
        print(f"  λ (lambda): {model.head.lmbda.item():.4f}")
        print(f"  γ (scale): {model.head.scale.item():.2f}")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        return
    
    print("\\n🎉 Simple example completed successfully!")
    print("\\nTo run a full experiment, use:")
    print("  python train_faim.py --dataset cotton80 --quick fast_training")


if __name__ == "__main__":
    try:
        simple_example()
    except KeyboardInterrupt:
        print("\\n⏹️ Example interrupted by user")
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
