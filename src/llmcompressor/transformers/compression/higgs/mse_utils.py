"""
Utilities for computing MSE between original and quantized weights.

This module provides functions to quantize weight tensors and compute
Mean Squared Error (MSE) for evaluating quantization quality.
"""

import torch
from compressed_tensors.quantization import (
    QuantizationScheme,
    fake_quantize,
    initialize_module_for_quantization,
)
from loguru import logger

from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    observe,
    update_qparams,
)


__all__ = ["compute_layer_mse", "quantize_weight"]


def quantize_weight(
    weight: torch.Tensor,
    scheme: QuantizationScheme,
    device: torch.device | str = None,
) -> torch.Tensor:
    """
    Quantize a weight tensor using the given quantization scheme.

    Uses the same quantization pipeline as model_free_ptq to ensure
    MSE calculations match actual quantization behavior.

    Args:
        weight: Original weight tensor (2D, shape [out_features, in_features])
        scheme: Quantization scheme defining how to quantize
        device: Device to perform quantization on (defaults to weight's device)

    Returns:
        Quantized weight tensor (fake-quantized, same shape and dtype as input)
    """
    if device is None:
        device = weight.device

    device = torch.device(device)

    # Validate weight shape
    if weight.ndim != 2:
        raise ValueError(
            f"Weight must be 2D (Linear weight), got shape {weight.shape}"
        )

    # Create a temporary Linear module to hold the weight
    out_features, in_features = weight.shape
    module = torch.nn.Linear(
        in_features, out_features, bias=False, device=device, dtype=weight.dtype
    )
    module.weight.data.copy_(weight.to(device))

    # Initialize quantization on the module
    # This adds weight_scale, weight_zero_point, etc. as buffers
    initialize_module_for_quantization(module, scheme, force_zero_point=False)

    # Calibrate: compute quantization parameters from the weight
    # This is the same process as model_free_ptq
    initialize_observer(module, "weight")
    apply_calibration_status(module)
    observe(module, base_name="weight")
    update_qparams(module, base_name="weight")
    freeze_module_quantization(module)

    # Get the computed quantization parameters
    # After calibration, the module has weight_scale and weight_zero_point
    scale = module.weight_scale
    zero_point = module.weight_zero_point if hasattr(module, "weight_zero_point") else None

    # Manually apply fake quantization using the computed parameters
    # This simulates what compress_module would do
    quantized_weight = fake_quantize(
        x=module.weight.data,
        scale=scale,
        zero_point=zero_point if zero_point is not None else torch.zeros_like(scale),
        args=scheme.weights,
    )

    return quantized_weight


def compute_layer_mse(
    weight: torch.Tensor,
    scheme: QuantizationScheme,
    device: torch.device | str = None,
) -> float:
    """
    Compute Mean Squared Error between original and quantized weight.

    Args:
        weight: Original weight tensor
        scheme: Quantization scheme to apply
        device: Device to perform computation on

    Returns:
        Scalar MSE value (float >= 0)
    """
    try:
        if device is None:
            device = weight.device

        # Ensure weight is on the target device
        weight = weight.to(device)

        # Quantize the weight using the scheme
        quantized = quantize_weight(weight, scheme, device)

        # Compute MSE: mean of squared differences
        mse = torch.mean((weight - quantized) ** 2).item()

        return float(mse)

    except Exception as e:
        # If quantization fails for this scheme, return inf
        # This will exclude it from ILP selection
        logger.warning(
            f"Failed to compute MSE for scheme {scheme}: {e}. "
            f"Returning inf to exclude from ILP."
        )
        return float('inf')
