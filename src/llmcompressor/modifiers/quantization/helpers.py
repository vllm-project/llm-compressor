"""Helpers for quantization formats and default weight observers."""

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


def is_mxfp4aN(scheme: QuantizationScheme) -> bool:
    """Return whether a scheme pairs MXFP4 weights with up to 8-bit activations."""
    activations = scheme.input_activations
    return is_mxfp4(scheme) and activations is not None and activations.num_bits <= 8


def get_default_weight_observer(
    scheme: QuantizationScheme, has_calibration_data: bool
) -> str:
    """Return the base default weight observer for a scheme."""
    if is_fp8(scheme):
        return "memoryless_minmax"
    if not has_calibration_data and is_mxfp4(scheme):
        return "memoryless_mse"
    return "imatrix_mse" if has_calibration_data else "memoryless_minmax"


def get_weight_observer_defaults(
    scheme: QuantizationScheme,
) -> dict[str, int | float | bool | str | None]:
    """Return base format-specific kwargs for the selected weight observer."""
    args = scheme.weights
    if args is None:
        return {}

    observer = args.observer
    if observer not in ("mse", "memoryless_mse", "imatrix_mse"):
        return {}

    # Search defaults
    defaults: dict[str, int | float | bool | str | None] = {
        "maxshrink": 0.95,
        "expand": 1.0,
        "grid": 200,
        "grid_type": "linear",
        "patience": 0,
    }
    if is_nvfp4(scheme):
        defaults.update(maxshrink=1 - 0.8 / 1.8, expand=1.8)
    elif is_mxfp4(scheme):
        defaults.update(maxshrink=1 - 1.0 / 64.0, grid_type="exponential")

    # Objective defaults
    if is_nvfp4(scheme):
        defaults.update(norm=3.0, magnitude_normalization=True)
    elif observer == "imatrix_mse" and is_mxfp4aN(scheme):
        defaults.update(norm=2, magnitude_normalization=False)
    else:
        defaults.update(norm=2.8, magnitude_normalization=True)

    return defaults
