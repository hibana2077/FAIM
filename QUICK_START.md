# FAIM: Finsler-α Information Manifold for Ultra-Fine-Grained Visual Classification

## Quick Start

The UFGVC classification pipeline with FAIM-Head is now ready! Here's how to use it:

### 1. Basic Training
```bash
# Train on UFGVC with default settings
python train_faim.py

# Train with custom config
python train_faim.py --config config/custom.yaml

# Quick training for testing
python train_faim.py --epochs 2 --batch_size 16
```

### 2. Simple Demo
```bash
# Run a minimal example
python example_simple.py
```

### 3. Programmatic Usage
```python
from models.faim_head import create_faim_model, FAIMTrainer
from utils.data_utils import create_dataloaders
from config.settings import get_quick_config

# Create model
config = get_quick_config('resnet18', 'ufgvc')
model = create_faim_model(config)

# Load data
train_loader, val_loader, test_loader = create_dataloaders(config)

# Train
trainer = FAIMTrainer(model, config)
trainer.train(train_loader, val_loader)
```

## Features ✅

- **FAIM-Head Implementation**: Complete implementation of Finsler-α Information Manifold
- **Three-Phase Training**: Warm-up → Tune-Metric → Fine-tune
- **Modular Design**: Easy to extend and customize
- **Multi-Backend Support**: Works with any timm model
- **Comprehensive Evaluation**: Accuracy, loss visualization, t-SNE, confusion matrix
- **Configuration System**: YAML configs and quick templates
- **Automatic Dataset Download**: Downloads UFGVC if not present
- **Robust Testing**: Full test suite validates all components

## Architecture

```
FAIM/
├── models/
│   ├── faim_head.py     # FAIM-Head implementation
│   └── trainer.py       # Multi-phase training
├── config/
│   └── settings.py      # Configuration management
├── utils/
│   ├── data_utils.py    # Data loading & augmentation
│   └── eval_utils.py    # Evaluation & visualization
├── dataset/
│   └── ufgvc.py        # UFGVC dataset (provided)
└── train_faim.py       # Main training script
```

## Validated ✅

All components have been tested and validated:
- ✅ FAIM-Head integration with timm models
- ✅ Three-phase training pipeline
- ✅ Data loading and augmentation
- ✅ Evaluation and metrics
- ✅ Configuration system
- ✅ End-to-end pipeline

You can now start experimenting with different models, datasets, and hyperparameters!
