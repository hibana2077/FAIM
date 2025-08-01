# FAIM: Finsler-α Information Manifold for UFGVC

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of **FAIM-Head** for Ultra-Fine-Grained Visual Classification (UFGVC) using Finsler manifold geometry.

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Simple Usage

```python
from models import create_faim_model

# Create model with FAIM-Head
model = create_faim_model(
    model_name='vit_base_patch16_224',
    num_classes=80,
    lambda_init=0.1
)
```

### Training

```bash
# Quick training
python train_faim.py --dataset cotton80 --quick fast_training

# Custom training
python train_faim.py --dataset soybean --lambda_init 0.15 --epochs 10 25 15
```

### Simple Demo

```bash
python example_simple.py
```

## 🏗️ Architecture

FAIM-Head replaces traditional Linear layers with manifold-based distance computation:

**F_x(v) = √(v^T Σ v) + λ |β^T v|**

This enables directional-dependent metrics that amplify subtle differences in ultra-fine-grained categories.

## 📊 Supported Datasets

- **cotton80**: Cotton classification (80 classes)
- **soybean**: Soybean classification  
- **soy_ageing_r1-r6**: Soybean aging datasets

All datasets are automatically downloaded when first used.

## 🎯 Three-Phase Training

1. **Warm-up**: Freeze metric parameters, train prototypes
2. **Tune-metric**: Unfreeze directional parameters
3. **Fine-tune**: Joint optimization of all parameters

## 📈 Features

- ✅ **FAIM-Head**: Novel Finsler manifold classification head
- ✅ **UFGVC Support**: Multiple agricultural datasets
- ✅ **timm Integration**: Compatible with 300+ pre-trained models
- ✅ **Comprehensive Evaluation**: FAIM-specific metrics and visualizations
- ✅ **Flexible Configuration**: Easy hyperparameter management

## 📚 Documentation

See [README_DETAILED.md](README_DETAILED.md) for comprehensive documentation.

## 🎛️ Key Parameters

- **λ (lambda_init)**: Directional component strength (0.05-0.2)
- **γ (scale_init)**: Temperature parameter (10-30)
- **full_sigma**: Use full vs diagonal Σ matrix

## 🔧 Project Structure

```
FAIM/
├── models/          # FAIM-Head and training
├── dataset/         # UFGVC datasets  
├── config/          # Configuration management
├── utils/           # Data and evaluation utilities
├── docs/            # Documentation
├── train_faim.py    # Main training script
└── example_simple.py # Simple demo
```

## 📄 Citation

```bibtex
@article{faim2024,
  title={FAIM: Finsler-α Information Manifold for Ultra-Fine-Grained Visual Classification},
  author={Your Name},
  year={2024}
}
```

## 📜 License

MIT License - see LICENSE file for details.