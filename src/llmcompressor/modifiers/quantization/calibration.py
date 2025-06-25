from typing import Any, Dict, Optional, Tuple

import torch
from compressed_tensors.quantization import (
    DynamicType,
    KVCacheScaleType,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.quantization.utils import is_kv_cache_quant_scheme
from compressed_tensors.utils import align_module_device, update_parameter_data
from loguru import logger
from torch.nn import Module

from llmcompressor.modifiers.quantization.cache import QuantizedKVParameterCache
from llmcompressor.observers import Observer
from llmcompressor.utils.helpers import getattr_chain

DEFAULT_MAXSHRINK = 0.20
DEFAULT_PATIENCE = 5
DEFAULT_AVERAGING_CONSTANT = 0.01
DEFAULT_GRID = 100.0
DEFAULT_NORM = 2.4

__all__ = [
    "initialize_observer",
    "update_weight_zp_scale",
    "calibrate_input_hook",
    "calibrate_output_hook",
    "calibrate_kv_cache_input_hook",
    "calibrate_kv_cache_output_hook",
    "initialize_quantized_kv_cache",
    "freeze_module_quantization",
    "apply_calibration_status",
    "reset_quantization_status",
    "update_weight_global_scale",
]


def initialize_observer(
    module: Module,
    base_name: str,
):
    """
    Initialize observer module and attach as submodule.
    The name of the observer is fetched from the quantization_args.
    The name is then used to load the observer from the registry and attached
    to the module. The name of the observer uses the base_name provided.

    :param module: torch.nn.Module that the observer is being attached to
    :param base_name: str used to name the observer attribute

    """

    arg_name = "weights" if base_name == "weight" else f"{base_name}_activations"
    quantization_scheme = getattr(module, "quantization_scheme", None)
    if not quantization_scheme:
        # no quantization scheme nothing to do
        return

    quantization_args = getattr(quantization_scheme, arg_name, None)
    # dont need observers for dynamic
    if quantization_args is not None and quantization_args.dynamic in (
        False,
        DynamicType.LOCAL,
    ):
        observer_kwargs = quantization_args.observer_kwargs or {}
        observer = Observer.load_from_registry(
            quantization_args.observer,
            quantization_args=quantization_args,
            averaging_constant=observer_kwargs.get(
                "averaging_constant", DEFAULT_AVERAGING_CONSTANT
            ),
            # used by mse observer only, will be ignored by minmax observer
            maxshrink=observer_kwargs.get("maxshrink", DEFAULT_MAXSHRINK),
            patience=observer_kwargs.get("patience", DEFAULT_PATIENCE),
            grid=observer_kwargs.get("grid", DEFAULT_GRID),
            norm=observer_kwargs.get("norm", DEFAULT_NORM),
        )
        module.register_module(f"{base_name}_observer", observer)


def call_observer(
    module: Module,
    base_name: str,
    value: Optional[torch.Tensor] = None,
    should_calculate_gparam: bool = False,
    should_calculate_qparams: bool = True,
):
    """
    Call a module's attached input/weight/output observer using a provided value.
    Update the module's scale and zp using the observer's return values.

    :param module: torch.nn.Module
    :param base_name: substring used to fetch the observer, scales, and zp
    :param value: torch.Tensor to be passed to the observer for activations. If
        base_name is "weight", then the module's weight tensor will be used
    """
    with align_module_device(module):
        if base_name == "weight":
            value = module.weight
            g_idx = getattr(module, "weight_g_idx", None)
        elif value is not None:
            g_idx = None
        else:
            raise ValueError(
                "Must provide a value to observe if not using weight observer"
            )

        observer = getattr(module, f"{base_name}_observer")

        if should_calculate_gparam:
            global_scale = observer(
                value,
                should_calculate_gparam=True,
            )
            update_parameter_data(module, global_scale, f"{base_name}_global_scale")
        else:
            global_scale = getattr(module, f"{base_name}_global_scale", None)

        if should_calculate_qparams:
            updated_scale, updated_zero_point = observer(
                value, g_idx=g_idx, global_scale=global_scale
            )
            update_parameter_data(module, updated_scale, f"{base_name}_scale")
            update_parameter_data(module, updated_zero_point, f"{base_name}_zero_point")


def update_weight_global_scale(module: Module):
    if getattr_chain(module, "quantization_scheme.weights", None) is None:
        return

    if (
        getattr_chain(module, "quantization_scheme.weights.strategy", None)
        != QuantizationStrategy.TENSOR_GROUP
    ):
        return

    call_observer(
        module,
        base_name="weight",
        should_calculate_gparam=True,
        should_calculate_qparams=False,
    )
    module.weight_observer.reset()


def update_weight_zp_scale(module: Module):
    """
    marks a layer as ready for calibration which activates observers
    to update scales and zero points on each forward pass

    apply to full model with `model.apply(update_weight_zp_scale)`

    :param module: module to set for calibration
    :param quantize_weights_upfront: whether to automatically
       run weight quantization at the start of calibration
    """
    if getattr_chain(module, "quantization_scheme.weights", None) is None:
        return

    if getattr(module, "quantization_status", None) != QuantizationStatus.CALIBRATION:
        logger.warning(
            "Attempting to calibrate weights of a module not in calibration mode"
        )

    call_observer(module=module, base_name="weight")


def calibrate_activations(module: Module, value: torch.Tensor, base_name: str):
    """
    Calibrate input or output activations by calling the a module's attached
    observer.

    :param module: torch.nn.Module
    :param base_name: substring used to fetch the observer, scales, and zp
    :param value: torch.Tensor to be passed to the observer

    """
    # If empty tensor, can't update zp/scale
    # Case for MoEs
    if value.numel() == 0:
        return

    quantization_scheme = getattr(module, "quantization_scheme", None)
    quantization_args = getattr(quantization_scheme, f"{base_name}_activations", None)

    calculate_qparams = True
    calculate_gparam = False

    if quantization_args is not None:
        if quantization_args.dynamic in (True, DynamicType.LOCAL):
            calculate_qparams = False
        if quantization_args.strategy == QuantizationStrategy.TENSOR_GROUP:
            calculate_gparam = True

    call_observer(
        module=module,
        base_name=base_name,
        value=value,
        should_calculate_gparam=calculate_gparam,
        should_calculate_qparams=calculate_qparams,
    )


def calibrate_input_hook(module: Module, args: Any):
    """
    Hook to calibrate input activations.
    Will call the observers to update the scales/zp before applying
    input QDQ in the module's forward pass.
    """
    args = args[0] if isinstance(args, tuple) else args
    calibrate_activations(module, value=args, base_name="input")


def calibrate_output_hook(module: Module, _args: Any, output: torch.Tensor):
    """
    Hook to calibrate output activations.
    Will call the observers to update the scales/zp before applying
    output QDQ.
    """
    calibrate_activations(
        module,
        value=output,
        base_name="output",
    )
    output = forward_quantize(
        module=module,
        value=output,
        base_name="output",
        args=module.quantization_scheme.output_activations,
    )
    return output


def calibrate_kv_cache_input_hook(
    module: Module, args: Any, kwargs: Dict[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Hook to update inputs to attention layers when running
    kv_cache quantization. Will update the passed in
    kv_cache to singleton QuantizedKVParameterCache.
    """
    kv_cache = getattr(module, "kv_cache")
    kwargs["past_key_value"] = kv_cache
    kwargs["use_cache"] = False
    return args, kwargs


def calibrate_kv_cache_output_hook(module: Module, _args: Any, _output: torch.Tensor):
    """
    Hook to update k_scale and v_scale parameters when running kv_cache quantization.
    """
    kv_cache = getattr(module, "kv_cache")
    k_scale = kv_cache.k_scales[module.layer_idx]
    v_scale = kv_cache.v_scales[module.layer_idx]
    update_parameter_data(module, k_scale, KVCacheScaleType.KEY.value)
    update_parameter_data(module, v_scale, KVCacheScaleType.VALUE.value)


def initialize_quantized_kv_cache(module: Module):
    """
    Initialize a quantized kv_cache on a module (analogous to initializing an observer)
    When a config specifying kv_cache quantization is applied to a model, the kv_cache
    args are redefined as the output_activations targeting attention modules.

    This function should be called on attention modules with output_activations
    """
    scheme: Optional[QuantizationScheme] = getattr(module, "quantization_scheme", None)
    existing_kv_cache = getattr(module, "kv_cache", None)

    if (
        scheme is None
        or not is_kv_cache_quant_scheme(scheme)
        or isinstance(existing_kv_cache, QuantizedKVParameterCache)
    ):
        return

    quantized_kv_cache = QuantizedKVParameterCache(scheme.output_activations)
    setattr(module, "kv_cache", quantized_kv_cache)


def apply_calibration_status(module: Module):
    scheme = getattr(module, "quantization_scheme", None)
    if not scheme:
        # no quantization scheme nothing to do
        return
    module.quantization_status = QuantizationStatus.CALIBRATION


def freeze_module_quantization(module: Module):
    """
    deletes observers when calibration is complete.

    apply to full model with `model.apply(freeze_module_quantization)`

    :param module: module to freeze quantization for
    """
    scheme = getattr(module, "quantization_scheme", None)
    if not scheme:
        # no quantization scheme nothing to do
        return

    if module.quantization_status == QuantizationStatus.FROZEN:
        # nothing to do, already frozen
        return

    # remove observers
    for name in ("input", "weight", "output"):
        obs_name = f"{name}_observer"
        if hasattr(module, obs_name):
            delattr(module, obs_name)

    # remove quantized kv_cache
    kv_cache = getattr(module, "kv_cache", None)
    if isinstance(kv_cache, QuantizedKVParameterCache):
        delattr(module, "kv_cache")

    module.quantization_status = QuantizationStatus.FROZEN


def reset_quantization_status(model: Module):
    for module in model.modules():
        if hasattr(module, "quantization_status"):
            delattr(module, "quantization_status")
