import torch
from compressed_tensors.quantization import (
    QuantizationScheme,
    initialize_module_for_quantization,
)

from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    observe,
    update_qparams,
)
from llmcompressor.modifiers.quantization.helpers import (
    get_default_weight_observer,
    get_weight_observer_defaults,
)
from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = [
    "initialize_quantized_linear",
    "validate_weight_for_quantization",
    "calibrate_weight",
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


def calibrate_weight(module: torch.nn.Linear):
    initialize_weight_observer(module)
    apply_calibration_status(module)
    observe(module, base_name="weight")
    update_qparams(module, base_name="weight")
    freeze_module_quantization(module)


def initialize_weight_observer(module: torch.nn.Linear):
    """Initialize a weight observer with model-free, no-calibration defaults."""
    scheme: QuantizationScheme = module.quantization_scheme
    args = scheme.weights
    if args is None:
        return

    if args.dynamic is not True and (
        args.observer is None or "observer" not in args.model_fields_set
    ):
        args.observer = get_default_weight_observer(
            scheme, has_calibration_data=False
        )

    observer_kwargs = get_weight_observer_defaults(scheme)
    observer_kwargs.update(args.observer_kwargs or {})
    initialize_observer(module, "weight", observer_kwargs=observer_kwargs)
