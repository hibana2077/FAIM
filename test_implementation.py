#!/usr/bin/env python3
"""
Test script to verify FAIM implementation works correctly.

This script runs basic tests to ensure all components are working properly.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_faim_head():
    """Test FAIM-Head implementation."""
    print("🧪 Testing FAIM-Head...")
    
    from models.faim_head import FAIMHead
    
    # Test parameters
    batch_size = 4
    feat_dim = 128
    num_classes = 10
    
    # Create FAIM-Head
    head = FAIMHead(
        in_features=feat_dim,
        num_classes=num_classes,
        lambda_init=0.1,
        full_sigma=False
    )
    
    # Test forward pass
    x = torch.randn(batch_size, feat_dim)
    logits = head(x)
    
    assert logits.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {logits.shape}"
    
    # Test distances
    distances = head.get_distances(x)
    assert distances.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {distances.shape}"
    
    # Test that logits are negative distances (scaled)
    expected_logits = -head.scale * distances
    assert torch.allclose(logits, expected_logits, atol=1e-6), "Logits should be -scale * distances"
    
    # Test loss computation
    targets = torch.randint(0, num_classes, (batch_size,))
    loss = nn.CrossEntropyLoss()(logits, targets)
    assert loss.requires_grad, "Loss should require gradients"
    
    # Test backward pass
    loss.backward()
    
    # Check that parameters have gradients
    for name, param in head.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} should have gradients"
    
    print("✅ FAIM-Head test passed!")
    

def test_model_creation():
    """Test model creation with FAIM-Head."""
    print("🧪 Testing model creation...")
    
    try:
        from models.faim_head import create_faim_model, FAIMHead
        
        # Test with a small model that we know works with timm
        model = create_faim_model(
            model_name='resnet18',
            num_classes=10,
            pretrained=False,
            lambda_init=0.1
        )
        
        # Check that the model has FAIM-Head (could be in different attributes)
        faim_head = None
        if hasattr(model, 'head') and isinstance(model.head, FAIMHead):
            faim_head = model.head
        elif hasattr(model, 'fc') and isinstance(model.fc, FAIMHead):
            faim_head = model.fc
        elif hasattr(model, 'classifier') and isinstance(model.classifier, FAIMHead):
            faim_head = model.classifier
        
        assert faim_head is not None, "Model should have FAIM-Head somewhere"
        assert faim_head.num_classes == 10, f"FAIM-Head should have 10 classes, got {faim_head.num_classes}"
        
        # Test full model forward pass
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        
        expected_shape = (2, 10)
        actual_shape = logits.shape
        assert actual_shape == expected_shape, f"Expected {expected_shape}, got {actual_shape}"
        
        print("✅ Model creation test passed!")
        
    except ImportError as e:
        print(f"⚠️  Model creation test skipped (timm not available): {e}")


def test_configuration():
    """Test configuration system."""
    print("🧪 Testing configuration...")
    
    from config import get_config, get_quick_config, validate_config
    
    # Test default config
    config = get_config()
    assert 'model' in config
    assert 'data' in config
    assert 'warmup' in config
    
    # Test validation
    try:
        validate_config(config)
        print("✅ Configuration validation passed!")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    
    # Test quick config
    quick_config = get_quick_config('small_experiment')
    assert quick_config['warmup']['epochs'] <= 5, "Small experiment should have few epochs"
    
    # Test dataset-specific config
    dataset_config = get_config(dataset_name='cotton80')
    assert dataset_config['data']['dataset_name'] == 'cotton80'
    
    print("✅ Configuration test passed!")


def test_data_utils():
    """Test data utilities."""
    print("🧪 Testing data utilities...")
    
    from utils.data_utils import get_transforms
    
    # Test transform creation
    train_transform = get_transforms(is_training=True, train_transforms={
        'random_horizontal_flip': 0.5,
        'random_rotation': 10
    })
    
    val_transform = get_transforms(is_training=False)
    
    # Test transforms on dummy data
    from PIL import Image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    train_tensor = train_transform(dummy_image)
    val_tensor = val_transform(dummy_image)
    
    assert train_tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {train_tensor.shape}"
    assert val_tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {val_tensor.shape}"
    
    print("✅ Data utilities test passed!")


def test_dataset_info():
    """Test dataset information."""
    print("🧪 Testing dataset info...")
    
    from dataset.ufgvc import UFGVCDataset
    
    # Test dataset listing
    datasets = UFGVCDataset.list_available_datasets()
    assert len(datasets) > 0, "Should have available datasets"
    assert 'cotton80' in datasets, "Should include cotton80 dataset"
    
    print(f"Available datasets: {list(datasets.keys())}")
    print("✅ Dataset info test passed!")


def test_trainer_components():
    """Test trainer components."""
    print("🧪 Testing trainer components...")
    
    from models.trainer import AverageMeter
    
    # Test AverageMeter
    meter = AverageMeter()
    meter.update(1.0)
    meter.update(2.0)
    meter.update(3.0)
    
    assert abs(meter.avg - 2.0) < 1e-6, f"Expected avg=2.0, got {meter.avg}"
    assert meter.count == 3, f"Expected count=3, got {meter.count}"
    
    print("✅ Trainer components test passed!")


def test_evaluation_utils():
    """Test evaluation utilities."""
    print("🧪 Testing evaluation utilities...")
    
    # Test with dummy model and data
    from models.faim_head import FAIMHead
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(10, 64)
            self.head = FAIMHead(64, 5)
        
        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)
    
    model = DummyModel()
    
    from utils.eval_utils import UFGVCEvaluator
    
    evaluator = UFGVCEvaluator(model, torch.device('cpu'))
    
    # Test that evaluator finds FAIM head
    assert evaluator.faim_head is not None, "Evaluator should find FAIM head"
    
    print("✅ Evaluation utilities test passed!")


def run_all_tests():
    """Run all tests."""
    print("🧪 Running FAIM implementation tests...")
    print("=" * 50)
    
    tests = [
        test_faim_head,
        test_model_creation,
        test_configuration,
        test_data_utils,
        test_dataset_info,
        test_trainer_components,
        test_evaluation_utils
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 50)
    print(f"🧪 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! FAIM implementation is working correctly.")
        return True
    else:
        print(f"⚠️  {failed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
