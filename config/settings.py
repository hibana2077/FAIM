"""
Configuration management for UFGVC training with FAIM-Head.

This module provides default configurations and utilities for managing
training hyperparameters across different phases and datasets.
"""

from typing import Dict, Any, Optional
import yaml
from pathlib import Path


# Default configuration for UFGVC training with FAIM-Head
DEFAULT_CONFIG = {
    # Model configuration
    'model': {
        'name': 'vit_base_patch16_224',  # timm model name
        'pretrained': True,
        'faim_head': {
            'lambda_init': 0.1,
            'scale_init': 10.0,
            'full_sigma': False,
            'eps': 1e-6
        }
    },
    
    # Data configuration  
    'data': {
        'dataset_name': 'cotton80',
        'root': './data',
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': True,
        'image_size': 224,
        'crop_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        # Data augmentation
        'train_transforms': {
            'random_horizontal_flip': 0.5,
            'random_rotation': 10,
            'color_jitter': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            },
            'random_erasing': 0.1
        }
    },
    
    # Trainer configuration
    'trainer': {
        'use_contrastive': False,
        'contrastive_weight': 0.1
    },
    
    # Phase 1: Warm-up (freeze metric parameters)
    'warmup': {
        'epochs': 10,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'scheduler_type': 'cosine',
        'save_best': True
    },
    
    # Phase 2: Tune-metric (unfreeze β, λ, gradually Σ)
    'tune_metric': {
        'epochs': 30,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'scheduler_type': 'cosine',
        'save_best': True
    },
    
    # Phase 3: Fine-tune (all parameters)
    'finetune': {
        'epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'scheduler_type': 'cosine',
        'save_best': True
    },
    
    # Device and logging
    'device': 'auto',  # 'auto', 'cpu', 'cuda', 'cuda:0', etc.
    'log_dir': './logs',
    'seed': 42,
    'mixed_precision': True
}


# Dataset-specific configurations
DATASET_CONFIGS = {
    'cotton80': {
        'model': {
            'faim_head': {
                'lambda_init': 0.1,
                'scale_init': 15.0
            }
        },
        'warmup': {
            'epochs': 10,
            'lr': 1e-3
        },
        'tune_metric': {
            'epochs': 25,
            'lr': 5e-4
        },
        'finetune': {
            'epochs': 15,
            'lr': 1e-4
        }
    },
    
    'soybean': {
        'model': {
            'faim_head': {
                'lambda_init': 0.15,
                'scale_init': 12.0
            }
        },
        'warmup': {
            'epochs': 8,
            'lr': 1e-3
        },
        'tune_metric': {
            'epochs': 30,
            'lr': 5e-4
        },
        'finetune': {
            'epochs': 20,
            'lr': 1e-4
        }
    },
    
    'soy_ageing_r1': {
        'model': {
            'faim_head': {
                'lambda_init': 0.08,
                'scale_init': 18.0
            }
        },
        'warmup': {
            'epochs': 12,
            'lr': 8e-4
        },
        'tune_metric': {
            'epochs': 35,
            'lr': 4e-4
        },
        'finetune': {
            'epochs': 25,
            'lr': 8e-5
        }
    },
    
    'soy_ageing_r3': {
        'model': {
            'faim_head': {
                'lambda_init': 0.12,
                'scale_init': 14.0
            }
        }
    },
    
    'soy_ageing_r4': {
        'model': {
            'faim_head': {
                'lambda_init': 0.1,
                'scale_init': 16.0
            }
        }
    },
    
    'soy_ageing_r5': {
        'model': {
            'faim_head': {
                'lambda_init': 0.09,
                'scale_init': 13.0
            }
        }
    },
    
    'soy_ageing_r6': {
        'model': {
            'faim_head': {
                'lambda_init': 0.11,
                'scale_init': 15.0
            }
        }
    }
}


# Model-specific configurations
MODEL_CONFIGS = {
    'vit_base_patch16_224': {
        'data': {
            'batch_size': 32,
            'image_size': 224,
            'crop_size': 224
        },
        'warmup': {
            'lr': 1e-3
        },
        'tune_metric': {
            'lr': 5e-4
        },
        'finetune': {
            'lr': 1e-4
        }
    },
    
    'vit_large_patch16_224': {
        'data': {
            'batch_size': 16,  # Smaller batch size for larger model
            'image_size': 224,
            'crop_size': 224
        },
        'warmup': {
            'lr': 8e-4
        },
        'tune_metric': {
            'lr': 4e-4
        },
        'finetune': {
            'lr': 8e-5
        }
    },
    
    'resnet50': {
        'data': {
            'batch_size': 64,
            'image_size': 224,
            'crop_size': 224
        },
        'warmup': {
            'lr': 1e-3
        },
        'tune_metric': {
            'lr': 5e-4
        },
        'finetune': {
            'lr': 1e-4
        }
    },
    
    'efficientnet_b4': {
        'data': {
            'batch_size': 32,
            'image_size': 380,
            'crop_size': 380
        },
        'warmup': {
            'lr': 8e-4
        },
        'tune_metric': {
            'lr': 4e-4
        },
        'finetune': {
            'lr': 8e-5
        }
    },
    
    'convnext_base': {
        'data': {
            'batch_size': 32,
            'image_size': 224,
            'crop_size': 224
        },
        'warmup': {
            'lr': 1e-3
        },
        'tune_metric': {
            'lr': 5e-4
        },
        'finetune': {
            'lr': 1e-4
        }
    }
}


def deep_merge_dict(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base_dict.copy()
    
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result


def get_config(
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    config_file: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get training configuration with dataset and model-specific adjustments.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'cotton80', 'soybean')
        model_name: Name of the model (e.g., 'vit_base_patch16_224')
        config_file: Path to custom configuration YAML file
        overrides: Dictionary of configuration overrides
        
    Returns:
        Complete configuration dictionary
    """
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Apply dataset-specific configuration
    if dataset_name and dataset_name in DATASET_CONFIGS:
        config = deep_merge_dict(config, DATASET_CONFIGS[dataset_name])
        config['data']['dataset_name'] = dataset_name
    
    # Apply model-specific configuration
    if model_name and model_name in MODEL_CONFIGS:
        config = deep_merge_dict(config, MODEL_CONFIGS[model_name])
        config['model']['name'] = model_name
    
    # Load from configuration file if provided
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            config = deep_merge_dict(config, file_config)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    # Apply overrides
    if overrides:
        config = deep_merge_dict(config, overrides)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def print_config(config: Dict[str, Any], title: str = "Configuration"):
    """Pretty print configuration.
    
    Args:
        config: Configuration dictionary
        title: Title for the printout
    """
    import json
    
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(json.dumps(config, indent=2, default=str))
    print(f"{'='*50}\n")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['model', 'data', 'warmup', 'tune_metric', 'finetune']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate model configuration
    model_config = config['model']
    if 'name' not in model_config:
        raise ValueError("Model name must be specified")
    
    # Validate data configuration
    data_config = config['data']
    required_data_keys = ['dataset_name', 'batch_size']
    for key in required_data_keys:
        if key not in data_config:
            raise ValueError(f"Missing required data configuration key: {key}")
    
    # Validate phase configurations
    for phase in ['warmup', 'tune_metric', 'finetune']:
        phase_config = config[phase]
        required_phase_keys = ['epochs', 'lr']
        for key in required_phase_keys:
            if key not in phase_config:
                raise ValueError(f"Missing required {phase} configuration key: {key}")
    
    return True


# Configuration templates for quick start
QUICK_START_CONFIGS = {
    'small_experiment': {
        'warmup': {'epochs': 3},
        'tune_metric': {'epochs': 5},
        'finetune': {'epochs': 2},
        'data': {'batch_size': 16}
    },
    
    'fast_training': {
        'warmup': {'epochs': 5},
        'tune_metric': {'epochs': 15},
        'finetune': {'epochs': 10},
        'data': {'batch_size': 64}
    },
    
    'high_quality': {
        'warmup': {'epochs': 15},
        'tune_metric': {'epochs': 40},
        'finetune': {'epochs': 30},
        'data': {'batch_size': 32}
    }
}


def get_quick_config(template: str, **kwargs) -> Dict[str, Any]:
    """Get a quick start configuration template.
    
    Args:
        template: Template name ('small_experiment', 'fast_training', 'high_quality')
        **kwargs: Additional arguments passed to get_config()
        
    Returns:
        Configuration dictionary
    """
    if template not in QUICK_START_CONFIGS:
        available = list(QUICK_START_CONFIGS.keys())
        raise ValueError(f"Template '{template}' not found. Available: {available}")
    
    template_config = QUICK_START_CONFIGS[template]
    return get_config(overrides=template_config, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Example configurations:")
    
    # Default configuration
    print("\n1. Default configuration:")
    default_config = get_config()
    print_config(default_config, "Default Config")
    
    # Dataset-specific configuration
    print("\n2. Cotton80 dataset configuration:")
    cotton_config = get_config(dataset_name='cotton80')
    print_config(cotton_config, "Cotton80 Config")
    
    # Model-specific configuration
    print("\n3. ViT-Large configuration:")
    vit_config = get_config(model_name='vit_large_patch16_224')
    print_config(vit_config, "ViT-Large Config")
    
    # Combined configuration
    print("\n4. Combined configuration (Soybean + EfficientNet):")
    combined_config = get_config(
        dataset_name='soybean',
        model_name='efficientnet_b4'
    )
    print_config(combined_config, "Combined Config")
    
    # Quick start template
    print("\n5. Quick start configuration:")
    quick_config = get_quick_config('fast_training', dataset_name='cotton80')
    print_config(quick_config, "Quick Start Config")
    
    # Save example configuration
    save_config(combined_config, './example_config.yaml')
    print("Example configuration saved to './example_config.yaml'")
    
    # Validate configuration
    try:
        validate_config(combined_config)
        print("✓ Configuration validation passed")
    except ValueError as e:
        print(f"✗ Configuration validation failed: {e}")
