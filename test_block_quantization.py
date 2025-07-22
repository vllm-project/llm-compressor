#!/usr/bin/env python3
"""
Test script to verify block quantization implementation works correctly.
"""

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.lifecycle.initialize import (
    initialize_module_for_quantization,
)

from llmcompressor.modifiers.quantization.calibration import initialize_observer


def test_block_quantization_observer():
    """Test that block quantization actually works with real tensors"""
    print("Testing block quantization observer...")
    
    # Create a module to quantize
    module = torch.nn.Linear(256, 128)  # Divisible by 128x128 blocks
    
    # Set up block quantization scheme
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            symmetric=True,
            strategy="block",
            block_structure=[128, 128],
        ),
    )
    
    initialize_module_for_quantization(module, scheme)
    initialize_observer(module, "weight")
    
    # Verify observer exists
    assert hasattr(module, "weight_observer"), "Observer not created"
    
    # Get the observer and test block quantization
    observer = module.weight_observer
    print(f"Weight shape: {module.weight.shape}")
    print(f"Block structure: {observer.quantization_args.block_structure}")
    
    # Call the observer to trigger block quantization
    scales, zero_points = observer(module.weight)
    
    print(f"Scales shape: {scales.shape}")
    print(f"Zero points shape: {zero_points.shape}")
    
    # For 256x128 weight with 128x128 blocks, we should get 2x1 blocks
    expected_shape = (2, 1)  # 256/128 = 2 rows, 128/128 = 1 col
    
    assert scales.shape == expected_shape, f"Expected scales shape {expected_shape}, got {scales.shape}"
    assert zero_points.shape == expected_shape, f"Expected zero_points shape {expected_shape}, got {zero_points.shape}"
    
    # Verify scales and zero points are reasonable
    assert torch.all(scales > 0), "All scales should be positive"
    assert torch.all(torch.isfinite(scales)), "All scales should be finite"
    assert torch.all(torch.isfinite(zero_points)), "All zero points should be finite"
    
    print("✅ Block quantization observer test passed!")


def test_block_structure_validation():
    """Test that block structure validation works"""
    print("Testing block structure validation...")
    
    module = torch.nn.Linear(100, 50)  # NOT divisible by 128
    
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            symmetric=True,
            strategy="block",
            block_structure=[128, 128],  # This should fail
        ),
    )
    
    initialize_module_for_quantization(module, scheme)
    initialize_observer(module, "weight")
    
    observer = module.weight_observer
    
    try:
        # This should raise a ValueError due to non-divisible dimensions
        scales, zero_points = observer(module.weight)
        assert False, "Should have raised ValueError for non-divisible dimensions"
    except ValueError as e:
        print(f"✅ Correctly caught validation error: {e}")
    
    print("✅ Block structure validation test passed!")


def test_different_block_sizes():
    """Test different block sizes work correctly"""
    print("Testing different block sizes...")
    
    # Test with 64x64 blocks on 256x128 tensor
    module = torch.nn.Linear(256, 128)
    
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            symmetric=True,
            strategy="block",
            block_structure=[64, 64],
        ),
    )
    
    initialize_module_for_quantization(module, scheme)
    initialize_observer(module, "weight")
    
    observer = module.weight_observer
    scales, zero_points = observer(module.weight)
    
    # For 256x128 weight with 64x64 blocks: 4x2 blocks
    expected_shape = (4, 2)  # 256/64 = 4 rows, 128/64 = 2 cols
    
    assert scales.shape == expected_shape, f"Expected scales shape {expected_shape}, got {scales.shape}"
    assert zero_points.shape == expected_shape, f"Expected zero_points shape {expected_shape}, got {zero_points.shape}"
    
    print(f"✅ 64x64 blocks produced {scales.shape} scales correctly!")


if __name__ == "__main__":
    print("Testing block quantization implementation...\n")
    
    test_block_quantization_observer()
    print()
    test_block_structure_validation()
    print()
    test_different_block_sizes()
    
    print("\n🎉 All block quantization tests passed! The implementation works correctly.") 