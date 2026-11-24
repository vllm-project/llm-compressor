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


def resolve_grid_type(grid_type: str | None, scheme: QuantizationScheme) -> str:
    """Resolve grid spacing from an explicit value or the quantization format."""
    if grid_type is None:
        return "exponential" if is_mxfp4(scheme) else "linear"
    if grid_type not in ("linear", "exponential"):
        raise ValueError(
            "grid_type must be None, 'linear', or 'exponential', " f"got {grid_type!r}"
        )
    return grid_type
