import torch
from compressed_tensors.compressors import BaseCompressor
from compressed_tensors.config.format import _get_quant_compression_format
from compressed_tensors.quantization import (
    QuantizationScheme,
    initialize_module_for_quantization,
)

from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    update_weight_global_scale,
    update_weight_zp_scale,
)
from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = [
    "initialize_quantized_linear",
    "validate_weight_for_quantization",
    "calibrate_global_scale",
    "calibrate_scale_zp",
    "compress_module",
]


def validate_weight_for_quantization(
    weight: torch.Tensor, scheme: QuantizationScheme, tensor_name: str
):
    if weight.ndim != 2:
        raise ValueError(
            f"Unable to quantize tensor `{tensor_name}`: expected 2D linear weight, "
            f"but got shape {tuple(weight.shape)}"
        )

    try:
        flatten_for_calibration(weight, "weight", scheme.weights)
    except Exception as exc:
        raise ValueError(f"Unable to quantize tensor `{tensor_name}`: {exc}") from exc


def initialize_quantized_linear(
    weight: torch.Tensor, scheme: QuantizationScheme, device: str | torch.device
) -> torch.nn.Module:
    out_features, in_features = weight.shape
    module = torch.nn.Linear(
        in_features, out_features, bias=False, device=device, dtype=weight.dtype
    )
    module.weight.data.copy_(weight)
    initialize_module_for_quantization(module, scheme, force_zero_point=False)

    return module


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
