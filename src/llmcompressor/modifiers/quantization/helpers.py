"""Helpers for detecting microscaled quantization formats."""

from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.utils.mxfp_utils import should_generate_mx_scales


def is_nvfp4(scheme: QuantizationScheme) -> bool:
    """Return whether a scheme uses the NVFP4 weight format."""
    weights = scheme.weights
    return (
        weights is not None
        and weights.num_bits == 4
        and weights.type == "float"
        and weights.group_size == 16
    )


def is_fp8(scheme: QuantizationScheme) -> bool:
    """Return whether a scheme uses 8-bit floating-point weights."""
    weights = scheme.weights
    return weights is not None and weights.num_bits == 8 and weights.type == "float"


def is_mxfp4(scheme: QuantizationScheme) -> bool:
    """Return whether a scheme uses the MXFP4 weight format."""
    weights = scheme.weights
    return (
        weights is not None
        and weights.num_bits == 4
        and weights.type == "float"
        and should_generate_mx_scales(weights)
    )


def is_mxfp4a8(scheme: QuantizationScheme) -> bool:
    """Return whether a scheme pairs MXFP4 weights with 8-bit activations."""
    activations = scheme.input_activations
    return is_mxfp4(scheme) and activations is not None and activations.num_bits == 8
