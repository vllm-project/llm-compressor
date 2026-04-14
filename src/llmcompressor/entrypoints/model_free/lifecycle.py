import torch
from compressed_tensors.quantization import (
    QuantizationScheme,
    initialize_module_for_quantization,
)

from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    observe_and_update_qparams,
)
from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = [
    "initialize_quantized_linear",
    "validate_weight_for_quantization",
    "calibrate_weights",
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


def calibrate_weights(module: torch.nn.Linear):
    initialize_observer(module, "weight")
    apply_calibration_status(module)
    observe_and_update_qparams(module, base_name="weight")
    freeze_module_quantization(module)
