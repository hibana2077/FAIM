"""
FAIM: Finsler-α Information Manifold for Ultra-Fine-Grained Visual Classification

Main package initialization.
"""

__version__ = "1.0.0"
__author__ = "FAIM Team"
__description__ = "Finsler-α Information Manifold for Ultra-Fine-Grained Visual Classification"

# Import main components for easy access
from .models import FAIMHead, create_faim_model, FAIMTrainer
from .config import get_config, get_quick_config
from .utils import create_dataloaders, evaluate_model_comprehensive

__all__ = [
    'FAIMHead',
    'create_faim_model', 
    'FAIMTrainer',
    'get_config',
    'get_quick_config',
    'create_dataloaders',
    'evaluate_model_comprehensive'
]
