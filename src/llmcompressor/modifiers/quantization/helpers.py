"""Helpers for detecting microscaled quantization formats."""

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.utils.mxfp_utils import should_generate_mx_scales


def is_nvfp4(args: QuantizationArgs) -> bool:
    """Return whether quantization args use the NVFP4 weight format."""
    return args.num_bits == 4 and args.type == "float" and args.group_size == 16


def is_mxfp4(args: QuantizationArgs) -> bool:
    """Return whether quantization args use the MXFP4 weight format."""
    return (
        args.num_bits == 4 and args.type == "float" and should_generate_mx_scales(args)
    )


def is_mxfp4a8(scheme: QuantizationScheme) -> bool:
    """Return whether a scheme pairs MXFP4 weights with 8-bit activations."""
    weights = scheme.weights
    activations = scheme.input_activations
    return (
        weights is not None
        and is_mxfp4(weights)
        and activations is not None
        and activations.num_bits == 8
    )
