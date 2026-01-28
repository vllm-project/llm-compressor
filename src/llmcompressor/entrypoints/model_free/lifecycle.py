from typing import Optional, Tuple

import torch
from compressed_tensors.compressors import BaseCompressor
from compressed_tensors.config.format import _get_quant_compression_format
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationStrategy,
    initialize_module_for_quantization,
)

from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    update_weight_global_scale,
    update_weight_zp_scale,
)
from llmcompressor.utils.block_quant_padding import pad_weight_for_block_quant

__all__ = [
    "initialize_quantized_linear",
    "calibrate_global_scale",
    "calibrate_scale_zp",
    "compress_module",
]


def initialize_quantized_linear(
    weight: torch.Tensor, scheme: QuantizationScheme, device: str | torch.device
) -> Tuple[torch.nn.Module, Optional[Tuple[int, int]]]:
    """
    Initialize a quantized linear module from a weight tensor.

    For block quantization strategies, pads the weight tensor to ensure
    dimensions are divisible by the block size.

    :param weight: Weight tensor of shape (out_features, in_features)
    :param scheme: Quantization scheme to apply
    :param device: Device to place the module on
    :return: Tuple of (module, original_shape) where original_shape is the
             shape before padding, or None if no padding was applied
    """
    original_shape = None

    # Pad weight for block quantization if needed
    if (
        scheme.weights is not None
        and scheme.weights.strategy == QuantizationStrategy.BLOCK
        and scheme.weights.block_structure is not None
    ):
        weight, original_shape = pad_weight_for_block_quant(
            weight, scheme.weights.block_structure
        )

    out_features, in_features = weight.shape
    module = torch.nn.Linear(
        in_features, out_features, bias=False, device=device, dtype=weight.dtype
    )
    module.weight.data.copy_(weight)
    initialize_module_for_quantization(module, scheme, force_zero_point=False)

    return module, original_shape


def calibrate_global_scale(module: torch.nn.Linear):
    initialize_observer(module, "weight")
    apply_calibration_status(module)
    update_weight_global_scale(module)
    freeze_module_quantization(module)


def calibrate_scale_zp(module: torch.nn.Linear):
    initialize_observer(module, "weight")
    apply_calibration_status(module)
    update_weight_zp_scale(module)
    freeze_module_quantization(module)


def compress_module(module: torch.nn.Linear):
    scheme: QuantizationScheme = getattr(module, "quantization_scheme")

    format = _get_quant_compression_format(scheme.input_activations, scheme.weights)
    scheme.format = format.value

    compressor = BaseCompressor.load_from_registry(format.value)
    data = compressor.compress_weight(
        module.weight,
        quantization_args=scheme.weights,
        scale=getattr(module, "weight_scale"),
        zero_point=getattr(module, "weight_zero_point", None),
        global_scale=getattr(module, "weight_global_scale", None),
    )

    # `compress_weight` is a messy api
    delattr(module, "weight")
    for key, value in data.items():
        if hasattr(module, key):
            getattr(module, key).data = value
        else:
            module.register_parameter(
                key, torch.nn.Parameter(value, requires_grad=False)
            )
