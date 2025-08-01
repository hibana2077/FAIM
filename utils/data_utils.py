"""
Data utilities for UFGVC training with FAIM-Head.

This module provides data loading, transformation, and augmentation utilities
optimized for ultra-fine-grained visual classification tasks.
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np
from typing import Dict, Tuple, Optional, Callable, List
from collections import Counter
import timm.data

try:
    from ..dataset.ufgvc import UFGVCDataset
except ImportError:
    from dataset.ufgvc import UFGVCDataset


def get_transforms(
    image_size: int = 224,
    crop_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    train_transforms: Optional[Dict] = None,
    is_training: bool = True
) -> transforms.Compose:
    """Create data transforms for training or evaluation.
    
    Args:
        image_size: Size to resize images to
        crop_size: Size to crop images to  
        mean: ImageNet normalization mean
        std: ImageNet normalization std
        train_transforms: Dictionary of training augmentation parameters
        is_training: Whether this is for training (applies augmentations)
        
    Returns:
        Composed transforms
    """
    if is_training and train_transforms:
        transform_list = []
        
        # Resize
        transform_list.append(transforms.Resize((image_size, image_size)))
        
        # Random crop with padding
        if crop_size < image_size:
            transform_list.append(transforms.RandomCrop(crop_size, padding=4))
        
        # Random horizontal flip
        if 'random_horizontal_flip' in train_transforms:
            flip_prob = train_transforms['random_horizontal_flip']
            if flip_prob > 0:
                transform_list.append(transforms.RandomHorizontalFlip(flip_prob))
        
        # Random rotation
        if 'random_rotation' in train_transforms:
            rotation_degrees = train_transforms['random_rotation']
            if rotation_degrees > 0:
                transform_list.append(transforms.RandomRotation(rotation_degrees))
        
        # Color jitter
        if 'color_jitter' in train_transforms:
            cj_params = train_transforms['color_jitter']
            transform_list.append(transforms.ColorJitter(
                brightness=cj_params.get('brightness', 0),
                contrast=cj_params.get('contrast', 0),
                saturation=cj_params.get('saturation', 0),
                hue=cj_params.get('hue', 0)
            ))
        
        # Random perspective
        if 'random_perspective' in train_transforms:
            perspective_prob = train_transforms['random_perspective']
            if perspective_prob > 0:
                transform_list.append(transforms.RandomPerspective(
                    distortion_scale=0.1, p=perspective_prob
                ))
        
        # Random affine
        if 'random_affine' in train_transforms:
            affine_params = train_transforms['random_affine']
            transform_list.append(transforms.RandomAffine(
                degrees=affine_params.get('degrees', 0),
                translate=affine_params.get('translate', None),
                scale=affine_params.get('scale', None),
                shear=affine_params.get('shear', None)
            ))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        # Random erasing (applied after normalization)
        if 'random_erasing' in train_transforms:
            erasing_prob = train_transforms['random_erasing']
            if erasing_prob > 0:
                transform_list.append(transforms.RandomErasing(
                    p=erasing_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3)
                ))
        
    else:
        # Evaluation transforms (no augmentation)
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    
    return transforms.Compose(transform_list)


def get_timm_transforms(
    model_name: str,
    is_training: bool = True,
    additional_augs: Optional[Dict] = None
) -> transforms.Compose:
    """Get transforms using timm's data configuration.
    
    Args:
        model_name: Name of the timm model
        is_training: Whether this is for training
        additional_augs: Additional augmentation parameters
        
    Returns:
        Composed transforms optimized for the specific model
    """
    try:
        import timm
        
        # Create a temporary model to get data config
        model = timm.create_model(model_name, pretrained=True)
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        
        if is_training:
            # Create training transforms with timm
            transform = timm.data.create_transform(**data_cfg, is_training=True)
            
            # Add additional augmentations if specified
            if additional_augs:
                aug_list = list(transform.transforms)
                
                # Insert additional augs before ToTensor
                tensor_idx = next(i for i, t in enumerate(aug_list) 
                                if isinstance(t, transforms.ToTensor))
                
                if 'mixup_alpha' in additional_augs or 'cutmix_alpha' in additional_augs:
                    # Note: Mixup/CutMix are typically applied during training loop
                    pass
                
                transform = transforms.Compose(aug_list)
        else:
            # Create evaluation transforms with timm
            transform = timm.data.create_transform(**data_cfg, is_training=False)
            
        del model  # Clean up
        return transform
        
    except ImportError:
        raise ImportError("timm is required for timm transforms")


def create_balanced_sampler(
    dataset: UFGVCDataset,
    num_samples_per_class: Optional[int] = None
) -> WeightedRandomSampler:
    """Create a weighted sampler for balanced training.
    
    Args:
        dataset: UFGVC dataset
        num_samples_per_class: Number of samples per class (None for natural balancing)
        
    Returns:
        WeightedRandomSampler for balanced sampling
    """
    # Get class distribution
    targets = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(targets)
    num_classes = len(class_counts)
    
    if num_samples_per_class is None:
        # Use inverse frequency weighting
        weights = [1.0 / class_counts[target] for target in targets]
    else:
        # Use fixed number of samples per class
        weights = [num_samples_per_class / class_counts[target] for target in targets]
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    return sampler


def create_dataloaders(
    dataset_name: str,
    root: str = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 224,
    crop_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    train_transforms: Optional[Dict] = None,
    use_timm_transforms: bool = False,
    model_name: Optional[str] = None,
    balanced_sampling: bool = False,
    download: bool = True
) -> Dict[str, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        dataset_name: Name of the UFGVC dataset
        root: Root directory for data storage
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        image_size: Size to resize images to
        crop_size: Size to crop images to
        mean: Normalization mean
        std: Normalization std
        train_transforms: Training augmentation parameters
        use_timm_transforms: Whether to use timm's transforms
        model_name: Model name for timm transforms
        balanced_sampling: Whether to use balanced sampling
        download: Whether to download dataset if not found
        
    Returns:
        Dictionary of data loaders {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    dataloaders = {}
    
    # Get available splits
    available_splits = UFGVCDataset.get_dataset_splits(dataset_name, root)
    if not available_splits:
        available_splits = ['train', 'val', 'test']
    
    for split in available_splits:
        try:
            # Determine if this is training split
            is_training = (split == 'train')
            
            # Create transforms
            if use_timm_transforms and model_name:
                transform = get_timm_transforms(model_name, is_training)
            else:
                transform = get_transforms(
                    image_size=image_size,
                    crop_size=crop_size,
                    mean=mean,
                    std=std,
                    train_transforms=train_transforms if is_training else None,
                    is_training=is_training
                )
            
            # Create dataset
            dataset = UFGVCDataset(
                dataset_name=dataset_name,
                root=root,
                split=split,
                transform=transform,
                download=download
            )
            
            # Create sampler for training if needed
            sampler = None
            shuffle = is_training
            if is_training and balanced_sampling:
                sampler = create_balanced_sampler(dataset)
                shuffle = False  # Don't shuffle when using sampler
            
            # Create data loader
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=is_training  # Drop last batch for training to avoid batch size issues
            )
            
            print(f"Created {split} dataloader: {len(dataset)} samples, {len(dataloaders[split])} batches")
            
        except ValueError as e:
            print(f"Warning: Could not create {split} dataloader: {e}")
            continue
    
    return dataloaders


def get_class_weights(dataset: UFGVCDataset) -> torch.Tensor:
    """Compute class weights for handling class imbalance.
    
    Args:
        dataset: UFGVC dataset
        
    Returns:
        Class weights tensor
    """
    # Count samples per class
    targets = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(targets)
    num_classes = len(class_counts)
    
    # Compute weights (inverse frequency)
    total_samples = len(targets)
    weights = torch.zeros(num_classes)
    
    for class_idx, count in class_counts.items():
        weights[class_idx] = total_samples / (num_classes * count)
    
    return weights


def analyze_dataset_statistics(
    dataset: UFGVCDataset,
    num_samples: int = 1000
) -> Dict[str, any]:
    """Analyze dataset statistics for better understanding.
    
    Args:
        dataset: UFGVC dataset
        num_samples: Number of samples to analyze
        
    Returns:
        Dictionary containing dataset statistics
    """
    import cv2
    from PIL import Image
    import io
    
    # Sample random indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Statistics to collect
    image_sizes = []
    brightness_values = []
    contrast_values = []
    class_distribution = Counter()
    
    for idx in indices:
        # Get raw image data
        row = dataset.data.iloc[idx]
        image_bytes = row['image']
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        # Image size
        image_sizes.append(image.size)
        
        # Convert to grayscale for brightness/contrast analysis
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Brightness (mean pixel value)
        brightness_values.append(np.mean(gray))
        
        # Contrast (standard deviation of pixel values)
        contrast_values.append(np.std(gray))
        
        # Class distribution
        class_distribution[row['label']] += 1
    
    # Compute statistics
    width_stats = {
        'mean': np.mean([size[0] for size in image_sizes]),
        'std': np.std([size[0] for size in image_sizes]),
        'min': min([size[0] for size in image_sizes]),
        'max': max([size[0] for size in image_sizes])
    }
    
    height_stats = {
        'mean': np.mean([size[1] for size in image_sizes]),
        'std': np.std([size[1] for size in image_sizes]),
        'min': min([size[1] for size in image_sizes]),
        'max': max([size[1] for size in image_sizes])
    }
    
    brightness_stats = {
        'mean': np.mean(brightness_values),
        'std': np.std(brightness_values),
        'min': min(brightness_values),
        'max': max(brightness_values)
    }
    
    contrast_stats = {
        'mean': np.mean(contrast_values),
        'std': np.std(contrast_values),
        'min': min(contrast_values),
        'max': max(contrast_values)
    }
    
    return {
        'dataset_name': dataset.dataset_name,
        'split': dataset.split,
        'total_samples': len(dataset),
        'num_classes': len(dataset.classes),
        'analyzed_samples': len(indices),
        'width_stats': width_stats,
        'height_stats': height_stats,
        'brightness_stats': brightness_stats,
        'contrast_stats': contrast_stats,
        'class_distribution': dict(class_distribution),
        'class_balance_ratio': max(class_distribution.values()) / min(class_distribution.values()) if class_distribution else 1.0
    }


def visualize_samples(
    dataset: UFGVCDataset,
    num_samples: int = 16,
    samples_per_row: int = 4,
    figsize: Tuple[int, int] = (12, 12),
    save_path: Optional[str] = None
):
    """Visualize random samples from the dataset.
    
    Args:
        dataset: UFGVC dataset
        num_samples: Number of samples to visualize
        samples_per_row: Number of samples per row
        figsize: Figure size
        save_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    
    # Sample random indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Calculate grid dimensions
    rows = (num_samples + samples_per_row - 1) // samples_per_row
    
    fig, axes = plt.subplots(rows, samples_per_row, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        row = i // samples_per_row
        col = i % samples_per_row
        
        # Get image and label
        row_data = dataset.data.iloc[idx]
        image_bytes = row_data['image']
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Plot
        axes[row, col].imshow(image)
        axes[row, col].set_title(f"Class: {row_data['class_name']}\nLabel: {row_data['label']}", fontsize=8)
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, rows * samples_per_row):
        row = i // samples_per_row
        col = i % samples_per_row
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def create_data_module(config: Dict) -> Dict[str, DataLoader]:
    """Create data module from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of data loaders
    """
    data_config = config['data']
    
    # Extract parameters
    dataset_name = data_config['dataset_name']
    root = data_config.get('root', './data')
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    pin_memory = data_config.get('pin_memory', True)
    image_size = data_config.get('image_size', 224)
    crop_size = data_config.get('crop_size', 224)
    mean = data_config.get('mean', [0.485, 0.456, 0.406])
    std = data_config.get('std', [0.229, 0.224, 0.225])
    train_transforms = data_config.get('train_transforms', None)
    balanced_sampling = data_config.get('balanced_sampling', False)
    
    # Check if we should use timm transforms
    use_timm_transforms = data_config.get('use_timm_transforms', False)
    model_name = config.get('model', {}).get('name', None)
    
    return create_dataloaders(
        dataset_name=dataset_name,
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        image_size=image_size,
        crop_size=crop_size,
        mean=mean,
        std=std,
        train_transforms=train_transforms,
        use_timm_transforms=use_timm_transforms,
        model_name=model_name,
        balanced_sampling=balanced_sampling
    )


if __name__ == "__main__":
    # Example usage
    print("Testing data utilities...")
    
    # Test transform creation
    train_augs = {
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
    
    train_transform = get_transforms(train_transforms=train_augs, is_training=True)
    val_transform = get_transforms(is_training=False)
    
    print(f"Train transform: {train_transform}")
    print(f"Val transform: {val_transform}")
    
    # Test dataloader creation
    try:
        dataloaders = create_dataloaders(
            dataset_name='cotton80',
            batch_size=16,
            train_transforms=train_augs,
            download=True
        )
        
        print(f"Created dataloaders: {list(dataloaders.keys())}")
        
        if 'train' in dataloaders:
            train_loader = dataloaders['train']
            
            # Test a batch
            for batch_images, batch_labels in train_loader:
                print(f"Batch shape: {batch_images.shape}")
                print(f"Label shape: {batch_labels.shape}")
                print(f"Label range: [{batch_labels.min()}, {batch_labels.max()}]")
                break
        
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
    
    print("Data utilities testing completed!")
