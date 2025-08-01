# FAIM: Finsler-α Information Manifold for Ultra-Fine-Grained Visual Classification

A PyTorch implementation of **FAIM-Head** (Finsler-α Information Manifold Classification Head) for Ultra-Fine-Grained Visual Classification (UFGVC). This repository provides a complete training pipeline with the novel FAIM-Head that uses directional-dependent distance metrics to amplify intra-class variations while maintaining inter-class separability.

## 🌟 Key Features

- **🎯 FAIM-Head**: Novel classification head using Randers-type Finsler manifold geometry
- **📊 UFGVC Datasets**: Support for multiple agricultural classification datasets
- **🔄 Three-Phase Training**: Warm-up → Tune-metric → Fine-tune strategy
- **📈 Comprehensive Evaluation**: FAIM-specific metrics and visualizations
- **⚙️ Flexible Configuration**: Easy hyperparameter management and model selection
- **🔧 timm Integration**: Seamless integration with timm model library

## 🏗️ Architecture

The FAIM-Head replaces traditional Linear classification layers with a manifold-based distance computation:

```
F_x(v) = √(v^T Σ v) + λ |β^T v|
```

Where:
- `Σ`: Positive definite Fisher information matrix approximation
- `β`: Directional 1-form vector  
- `λ`: Weight controlling directional component

This enables **directional-dependent metrics** that can amplify subtle differences in ultra-fine-grained categories.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd FAIM

# Install dependencies
pip install -r requirements.txt
```

### Simple Example

```python
from models import create_faim_model
from config import get_quick_config
from utils import create_data_module

# Get configuration
config = get_quick_config('fast_training', dataset_name='cotton80')

# Create data loaders
dataloaders = create_data_module(config)

# Create model with FAIM-Head
model = create_faim_model(
    model_name='vit_base_patch16_224',
    num_classes=80,
    lambda_init=0.1
)

# The model now has a FAIM-Head instead of standard Linear layer!
```

### Training

#### Quick Training
```bash
# Fast training with default settings
python train_faim.py --dataset cotton80 --quick fast_training

# Small experiment for testing
python train_faim.py --dataset soybean --quick small_experiment

# High quality training
python train_faim.py --dataset cotton80 --quick high_quality --model vit_large_patch16_224
```

#### Custom Training
```bash
# Custom hyperparameters
python train_faim.py \
    --dataset cotton80 \
    --model vit_base_patch16_224 \
    --lambda_init 0.15 \
    --scale_init 12.0 \
    --epochs 10 25 15 \
    --batch_size 32

# With configuration file
python train_faim.py --config configs/cotton80_vit.yaml
```

### Simple Demo

```bash
# Run a simple demonstration
python example_simple.py
```

## 📊 Supported Datasets

The pipeline supports multiple UFGVC datasets:

| Dataset | Classes | Description |
|---------|---------|-------------|
| `cotton80` | 80 | Cotton classification with 80 varieties |
| `soybean` | Variable | Soybean classification dataset |
| `soy_ageing_r1` to `soy_ageing_r6` | Variable | Soybean aging datasets (rounds 1-6) |

Datasets are automatically downloaded when first used.

```bash
# List all available datasets
python train_faim.py --list_datasets
```

## ⚙️ Configuration

### Quick Start Templates

```python
from config import get_quick_config

# Small experiment (fast testing)
config = get_quick_config('small_experiment', dataset_name='cotton80')

# Fast training (balanced speed/quality)  
config = get_quick_config('fast_training', dataset_name='soybean')

# High quality (best results)
config = get_quick_config('high_quality', dataset_name='cotton80')
```

### Custom Configuration

```python
from config import get_config

# Dataset and model specific
config = get_config(
    dataset_name='cotton80',
    model_name='vit_base_patch16_224'
)

# With custom overrides
config = get_config(
    dataset_name='soybean',
    overrides={
        'model': {'faim_head': {'lambda_init': 0.2}},
        'warmup': {'epochs': 15, 'lr': 8e-4}
    }
)
```

### Configuration File

```yaml
# config.yaml
model:
  name: vit_base_patch16_224
  faim_head:
    lambda_init: 0.1
    scale_init: 10.0
    full_sigma: false

data:
  dataset_name: cotton80
  batch_size: 32
  
warmup:
  epochs: 10
  lr: 1e-3

tune_metric:
  epochs: 25
  lr: 5e-4
  
finetune:
  epochs: 15
  lr: 1e-4
```

## 🎯 Training Strategy

The pipeline implements a **three-phase training approach**:

### Phase 1: Warm-up (5-15 epochs)
- **Freeze**: Metric parameters (Σ, β, λ)
- **Train**: Class prototypes (μ_k) and backbone
- **Goal**: Stabilize prototype positions and feature distributions

### Phase 2: Tune-metric (20-40 epochs)  
- **Unfreeze**: β and λ parameters
- **Gradually unfreeze**: Σ matrix
- **Goal**: Learn optimal directional-dependent metrics

### Phase 3: Fine-tune (15-30 epochs)
- **Train**: All parameters jointly
- **Goal**: Final optimization of the complete system

```python
from models import FAIMTrainer

trainer = FAIMTrainer(model, device)

# Phase 1: Warm-up
trainer.train_phase('warmup', train_loader, val_loader, epochs=10)

# Phase 2: Tune-metric
trainer.train_phase('tune_metric', train_loader, val_loader, epochs=25)

# Phase 3: Fine-tune  
trainer.train_phase('finetune', train_loader, val_loader, epochs=15)
```

## 📈 Evaluation

### Comprehensive Evaluation

```python
from utils import evaluate_model_comprehensive

results = evaluate_model_comprehensive(
    model=model,
    dataloader=test_loader,
    device=device,
    class_names=class_names,
    save_dir='./evaluation_results'
)

print(f"Accuracy: {results['standard_metrics']['accuracy']:.3f}")
print(f"FAIM Distance Accuracy: {results['faim_metrics']['distance_accuracy']:.3f}")
```

### FAIM-Specific Metrics

The evaluation includes specialized metrics for FAIM analysis:

- **Distance-based accuracy**: Classification using raw FAIM distances
- **Margin analysis**: Distribution of classification margins
- **Parameter analysis**: λ, γ, and Σ eigenvalues
- **Prototype distances**: Inter-class prototype separations
- **Feature visualization**: t-SNE plots of learned representations

### Visualization

```python
from utils import UFGVCEvaluator

evaluator = UFGVCEvaluator(model, device, class_names)
results = evaluator.evaluate_comprehensive(test_loader)

# Plot confusion matrix
evaluator.plot_confusion_matrix(results['standard_metrics']['confusion_matrix'])

# Plot FAIM margins
evaluator.plot_margin_distribution(results['faim_metrics']['raw_margins'])

# Plot feature t-SNE
evaluator.plot_feature_tsne(results['features'], results['targets'])
```

## 🏛️ Project Structure

```
FAIM/
├── 📁 models/               # FAIM-Head implementation and training
│   ├── faim_head.py        # Core FAIM-Head module
│   ├── trainer.py          # Training pipeline
│   └── __init__.py
├── 📁 dataset/             # UFGVC dataset handling
│   ├── ufgvc.py           # Dataset classes and utilities
│   └── __init__.py
├── 📁 config/              # Configuration management
│   ├── settings.py        # Configuration utilities
│   └── __init__.py
├── 📁 utils/               # Data and evaluation utilities
│   ├── data_utils.py      # Data loading and augmentation
│   ├── eval_utils.py      # Evaluation and visualization
│   └── __init__.py
├── 📁 docs/                # Documentation
│   ├── FAIM_head.md       # Detailed FAIM-Head documentation
│   ├── method.md          # Integration methods
│   ├── timm.md            # timm usage notes
│   └── abs.md             # Research abstract
├── 📁 scripts/             # Training scripts
│   ├── run_ccfso.sh       # Example training scripts
│   └── test.sh
├── train_faim.py           # Main training script
├── example_simple.py       # Simple usage example
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🔬 Research Background

FAIM-Head addresses the key challenges in Ultra-Fine-Grained Visual Classification:

1. **Small inter-class differences**: Traditional Euclidean/Riemannian metrics struggle with subtle visual differences
2. **Large intra-class variations**: Need to accommodate natural variation within categories  
3. **Directional sensitivity**: Important visual differences may be directional in feature space

The Finsler-α manifold approach provides:

- **Direction-dependent metrics**: Different sensitivity along different feature directions
- **Amplified discriminability**: Larger effective distances along discriminative directions
- **Maintained separability**: Preserved or increased inter-class margins

## 📚 Mathematical Foundation

### FAIM Metric Definition

For point `x` and tangent vector `v`:

```
F_x(v) = √(v^T Σ v) + λ |β^T v|
```

### Geodesic Distance (Closed Form)

For constant Σ and β:

```
d_F(p,q) = √((q-p)^T Σ (q-p)) + λ |β·(q-p)|
```

### Margin Amplification Theorem

Under FAIM, the classification margin satisfies:

```
m_F ≥ m_E + λ |β·w| / ||w||_2
```

Where `m_E` is the Euclidean margin and `w` is the class separation direction.

## 🎛️ Hyperparameters

### Key FAIM Parameters

- **λ (lambda_init)**: Controls directional component strength
  - Range: 0.05 - 0.2
  - Higher values → more directional sensitivity
  
- **γ (scale_init)**: Temperature parameter for softmax
  - Range: 10 - 30
  - Higher values → sharper probability distributions
  
- **full_sigma**: Use full Σ matrix vs diagonal
  - `False`: Faster, fewer parameters
  - `True`: More expressive, slower

### Training Parameters

- **Learning rates**: Typically 1e-3 → 5e-4 → 1e-4 across phases
- **Epochs**: 10 → 25 → 15 (warmup → tune → finetune)
- **Batch size**: 16-64 depending on model and GPU memory

## 🔧 Advanced Usage

### Custom FAIM-Head

```python
from models import FAIMHead

# Create custom FAIM-Head
faim_head = FAIMHead(
    in_features=768,
    num_classes=100, 
    lambda_init=0.15,
    scale_init=12.0,
    full_sigma=True  # Use full Σ matrix
)

# Replace existing head
model.head = faim_head
```

### Custom Training Loop

```python
from models import FAIMTrainer, FAIMContrastiveLoss

# Create trainer with contrastive loss
trainer = FAIMTrainer(
    model, 
    device,
    use_contrastive=True,
    contrastive_weight=0.1
)

# Custom phase training
for phase in ['warmup', 'tune_metric', 'finetune']:
    trainer.train_phase(
        phase=phase,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=phase_epochs[phase],
        lr=phase_lrs[phase]
    )
```

### Multi-Dataset Training

```python
from utils import create_dataloaders

# Train on multiple datasets
datasets = ['cotton80', 'soybean', 'soy_ageing_r1']

for dataset_name in datasets:
    dataloaders = create_dataloaders(dataset_name=dataset_name)
    # Train model on each dataset...
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   # Reduce batch size
   python train_faim.py --batch_size 16
   
   # Use gradient accumulation (modify config)
   ```

2. **Slow training**
   ```bash
   # Reduce number of workers if I/O bound
   python train_faim.py --num_workers 2
   
   # Use diagonal Σ instead of full matrix
   python train_faim.py --no-full_sigma
   ```

3. **Poor convergence**
   ```bash
   # Try different λ value
   python train_faim.py --lambda_init 0.05
   
   # Increase warm-up epochs
   python train_faim.py --epochs 15 25 15
   ```

### Debug Mode

```bash
# Quick debug run with small dataset
python train_faim.py --debug --dataset cotton80

# Dry run (load config and data, but don't train)
python train_faim.py --dry_run --dataset soybean
```

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{faim2024,
  title={FAIM: Finsler-α Information Manifold for Ultra-Fine-Grained Visual Classification},
  author={Your Name},
  journal={Conference/Journal},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **timm library**: For providing excellent pre-trained models
- **UFGVC benchmark**: For standardized evaluation datasets
- **Finsler geometry research**: For mathematical foundations
