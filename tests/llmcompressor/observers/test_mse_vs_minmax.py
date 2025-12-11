"""
Test to verify that MSE observer performs equal to or better than MinMax observer
on various tensor distributions, including normal distributions (similar to real weights)
and actual model weights.

This test checks that the quantization error (MSE) from using MSE observer
is less than or equal to the error from using MinMax observer.
"""

import pytest
import torch
from compressed_tensors.quantization import fake_quantize
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.observers import Observer


def _create_base_quantization_args(num_bits, strategy, symmetric, group_size):
    """Helper to create base QuantizationArgs without observer field."""
    return QuantizationArgs(
        num_bits=num_bits,
        strategy=strategy,
        symmetric=symmetric,
        group_size=group_size,
    )


def _run_observer_test(tensor, observer_name, strategy, symmetric, num_bits, group_size, module=None):
    """
    Helper function to run observer and compute quantization error.
    
    Returns: (scale, zero_point, quantized_tensor, mse, global_scale)
    """
    weights = _create_base_quantization_args(num_bits, strategy, symmetric, group_size)
    weights.observer = observer_name
    
    observer = Observer.load_from_registry(
        observer_name, base_name="weight", args=weights, module=module
    )
    
    global_scale = None
    if strategy == "tensor_group" and module is not None:
        global_scale = observer.get_global_scale(tensor)
        module.weight_global_scale = global_scale
    
    scale, zero_point = observer(tensor)
    
    # Sanity check: scales should be non-negative
    assert (scale >= 0).all(), "Scale values should be non-negative"
    
    weights_clean = _create_base_quantization_args(num_bits, strategy, symmetric, group_size)
    quantized = fake_quantize(
        tensor, scale, zero_point, weights_clean,
        global_scale=global_scale if strategy == "tensor_group" else None
    )
    mse = torch.nn.functional.mse_loss(quantized, tensor)
    
    return scale, zero_point, quantized, mse, global_scale


def _assert_mse_comparison(mse_mse, minmax_mse, strategy, symmetric, is_real_weights=False):
    """
    Assert MSE observer performance with appropriate slack.
    
    For tensor+symmetric: strict assertion (MSE should be better)
    For others: allow slack (10% for synthetic, 20% for real weights)
    Also add epsilon to handle cases where minmax_mse is near 0.
    """
    epsilon = 1e-8
    slack = 1.20 if is_real_weights else 1.10
    
    if strategy == "tensor" and symmetric:
        # Cases where MSE SHOULD be better
        assert mse_mse <= minmax_mse + epsilon, (
            f"MSE observer performed worse than MinMax observer!\n"
            f"Strategy: {strategy}, Symmetric: {symmetric}\n"
            f"MinMax MSE: {minmax_mse.item():.6e}\n"
            f"MSE Observer MSE: {mse_mse.item():.6e}\n"
            f"Difference: {(mse_mse - minmax_mse).item():.6e}"
        )
    else:
        # Not guaranteed, but ensure not catastrophically worse
        assert mse_mse <= minmax_mse * slack + epsilon, (
            f"MSE observer performed significantly worse than MinMax observer!\n"
            f"Strategy: {strategy}, Symmetric: {symmetric}\n"
            f"MinMax MSE: {minmax_mse.item():.6e}\n"
            f"MSE Observer MSE: {mse_mse.item():.6e}\n"
            f"Difference: {(mse_mse - minmax_mse).item():.6e}\n"
            f"Ratio: {(mse_mse / (minmax_mse + epsilon)).item():.4f}x"
        )


@pytest.mark.parametrize(
    "strategy,symmetric,num_bits",
    [
        ("tensor", True, 8),
        ("tensor", False, 8),
        ("channel", True, 8),
        ("channel", False, 8),
        ("tensor_group", True, 4),
        ("tensor_group", False, 4),
        ("channel", True, 4),
        ("channel", False, 4),
    ],
)
@pytest.mark.parametrize(
    "std",
    [0.05, 0.2, 1.0],
    ids=["narrow", "medium", "wide"],
)
def test_mse_vs_minmax_on_random_tensor(strategy, symmetric, num_bits, std):
    """
    Test that MSE observer produces quantization error <= MinMax observer
    on random tensors with normal distribution (similar to real model weights).
    
    Real model weights typically follow a normal distribution with:
    - Mean near 0
    - Standard deviation around 0.02-0.1 for initialized weights
    - Range roughly [-0.5, 0.5] for most layers
    
    Testing with different std values exposes cases where MinMax performs poorly
    on wide or heavy-tailed distributions, where MSE should shine.
    """
    # Generate random tensor with normal distribution similar to real weights
    torch.manual_seed(42)
    # Use different std values to test various distribution widths
    tensor = torch.randn(128, 256) * std  # Normal distribution with specified std
    
    group_size = 32 if strategy == "tensor_group" else None
    
    # Create separate modules for tensor_group to avoid shared mutable state
    module_minmax = None
    module_mse = None
    if strategy == "tensor_group":
        module_minmax = torch.nn.Linear(256, 128)
        module_minmax.weight.data = tensor
        module_mse = torch.nn.Linear(256, 128)
        module_mse.weight.data = tensor
    
    # Test with MinMax observer
    _, _, _, minmax_mse, _ = _run_observer_test(
        tensor, "memoryless_minmax", strategy, symmetric, num_bits, group_size, module_minmax
    )
    
    # Test with MSE observer
    _, _, _, mse_mse, _ = _run_observer_test(
        tensor, "memoryless_mse", strategy, symmetric, num_bits, group_size, module_mse
    )
    
    # Assert with appropriate slack for synthetic data
    _assert_mse_comparison(mse_mse, minmax_mse, strategy, symmetric, is_real_weights=False)


@pytest.mark.parametrize(
    "tensor_shape",
    [
        (64, 128),
        (128, 256),
        (256, 512),
        (32, 64, 128),  # 3D tensor
    ],
)
def test_mse_vs_minmax_various_shapes(tensor_shape):
    """
    Test MSE vs MinMax on tensors of various shapes with normal distribution.
    Uses realistic weight distribution parameters.
    """
    torch.manual_seed(42)
    # Use realistic weight distribution: mean=0, std=0.05
    tensor = torch.randn(*tensor_shape) * 0.05
    
    # MinMax
    _, _, _, minmax_mse, _ = _run_observer_test(
        tensor, "memoryless_minmax", "channel", True, 8, None, None
    )
    
    # MSE
    _, _, _, mse_mse, _ = _run_observer_test(
        tensor, "memoryless_mse", "channel", True, 8, None, None
    )
    
    # Channel quantization: MSE not guaranteed better, allow 10% slack
    _assert_mse_comparison(mse_mse, minmax_mse, "channel", True, is_real_weights=False)


def test_mse_vs_minmax_extreme_values():
    """Test MSE vs MinMax on tensors with extreme values."""
    torch.manual_seed(42)
    
    # Test with very small values
    tensor_small = torch.randn(64, 128) * 0.01
    # Test with very large values
    tensor_large = torch.randn(64, 128) * 100.0
    # Test with skewed distribution
    tensor_skewed = torch.cat([
        torch.randn(64, 100) * 0.1,
        torch.randn(64, 28) * 10.0
    ], dim=1)
    
    for tensor, name in [
        (tensor_small, "small"),
        (tensor_large, "large"),
        (tensor_skewed, "skewed"),
    ]:
        # MinMax
        _, _, _, minmax_mse, _ = _run_observer_test(
            tensor, "memoryless_minmax", "channel", True, 8, None, None
        )
        
        # MSE
        _, _, _, mse_mse, _ = _run_observer_test(
            tensor, "memoryless_mse", "channel", True, 8, None, None
        )
        
        # Channel quantization: MSE not guaranteed better, allow 10% slack
        _assert_mse_comparison(mse_mse, minmax_mse, "channel", True, is_real_weights=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    "strategy,symmetric,num_bits",
    [
        ("channel", True, 8),
        ("channel", False, 8),
        ("tensor_group", True, 4),
        ("tensor_group", False, 4),
    ],
)
def test_mse_vs_minmax_on_real_model_weights(strategy, symmetric, num_bits):
    """
    Test that MSE observer produces quantization error <= MinMax observer
    on actual model weights from a real neural network.
    
    This test loads weights from a small model to verify observer behavior
    on real weight distributions, which may differ from synthetic data.
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        pytest.skip("transformers not available")

    # Use a small, publicly available model for testing
    model_id = "nm-testing/tinysmokellama-3.2"
    
    try:
        # Load model and extract a weight tensor
        # Use no_grad context to avoid unnecessary gradient computation
        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float32
            )
            
            # Get a representative weight tensor (e.g., from first Linear layer)
            weight_tensor = None
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and weight_tensor is None:
                    weight_tensor = module.weight.data.clone()
                    break
        
        if weight_tensor is None:
            pytest.skip("No Linear layer found in model")
        
        # Flatten or reshape to 2D if needed for testing
        if weight_tensor.dim() > 2:
            weight_tensor = weight_tensor.view(-1, weight_tensor.shape[-1])
        elif weight_tensor.dim() == 1:
            weight_tensor = weight_tensor.unsqueeze(0)
        
        # Limit size for faster testing
        if weight_tensor.shape[0] > 512:
            weight_tensor = weight_tensor[:512, :]
        if weight_tensor.shape[1] > 512:
            weight_tensor = weight_tensor[:, :512]
        
    except Exception as e:
        pytest.skip(f"Could not load model {model_id}: {e}")
    
    group_size = 32 if strategy == "tensor_group" else None
    
    # Create separate modules for tensor_group to avoid shared mutable state
    module_minmax = None
    module_mse = None
    if strategy == "tensor_group":
        module_minmax = torch.nn.Linear(weight_tensor.shape[1], weight_tensor.shape[0])
        module_minmax.weight.data = weight_tensor
        module_mse = torch.nn.Linear(weight_tensor.shape[1], weight_tensor.shape[0])
        module_mse.weight.data = weight_tensor
    
    # Test with MinMax observer
    _, _, _, minmax_mse, _ = _run_observer_test(
        weight_tensor, "memoryless_minmax", strategy, symmetric, num_bits, group_size, module_minmax
    )
    
    # Test with MSE observer
    _, _, _, mse_mse, _ = _run_observer_test(
        weight_tensor, "memoryless_mse", strategy, symmetric, num_bits, group_size, module_mse
    )
    
    # For channel and tensor_group strategies, MSE is not guaranteed to be better
    # Allow 20% slack for real model weights (more structure & extreme channels)
    _assert_mse_comparison(mse_mse, minmax_mse, strategy, symmetric, is_real_weights=True)

