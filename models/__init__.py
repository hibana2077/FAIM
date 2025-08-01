"""
Models package for UFGVC with FAIM-Head.

This package provides:
- FAIMHead: Finsler-α Information Manifold classification head
- FAIMTrainer: Comprehensive training pipeline
- Model creation utilities
"""

from .faim_head import FAIMHead, FAIMContrastiveLoss, create_faim_model
from .trainer import FAIMTrainer, train_full_pipeline, AverageMeter

__all__ = [
    'FAIMHead',
    'FAIMContrastiveLoss', 
    'create_faim_model',
    'FAIMTrainer',
    'train_full_pipeline',
    'AverageMeter'
]
