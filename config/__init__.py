"""
Config package for UFGVC training configuration management.
"""

from .settings import (
    get_config,
    save_config, 
    print_config,
    validate_config,
    get_quick_config,
    DEFAULT_CONFIG,
    DATASET_CONFIGS,
    MODEL_CONFIGS,
    QUICK_START_CONFIGS
)

__all__ = [
    'get_config',
    'save_config',
    'print_config', 
    'validate_config',
    'get_quick_config',
    'DEFAULT_CONFIG',
    'DATASET_CONFIGS',
    'MODEL_CONFIGS',
    'QUICK_START_CONFIGS'
]
