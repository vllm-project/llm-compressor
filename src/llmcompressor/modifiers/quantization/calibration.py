from typing import Any, Optional

import torch
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationStatus,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.utils import align_module_device, update_offload_parameter
from loguru import logger
from torch.nn import Module

from llmcompressor.observers import Observer
from llmcompressor.utils.helpers import getattr_chain

DEFAULT_MAXSHRINK = 0.20
DEFAULT_PATIENCE = 5
DEFAULT_AVERAGING_CONSTANT = 0.01
DEFAULT_GRID = 100.0
DEFAULT_NORM = 2.4
ALL_OBSERVER_BASE_NAMES = {"input", "weight", "output", "q", "k", "v"}

__all__ = [
    "initialize_observer",
    "update_weight_zp_scale",
    "calibrate_input_hook",
    "calibrate_query_hook",
    "calibrate_key_hook",
    "calibrate_value_hook",
    "calibrate_output_hook",
    "freeze_module_quantization",
    "apply_calibration_status",
    "reset_quantization_status",
    "update_weight_global_scale",
]


def initialize_observer(module: Module, base_name: str):
    """
    Initialize observer module and attach as submodule.
    The name of the observer is fetched from the quantization_args.
    The name is then used to load the observer from the registry and attached
    to the module. The name of the observer uses the base_name provided.

    :param module: torch.nn.Module that the observer is being attached to
    :param base_name: str used to name the observer attribute

    """
    # resolve arg name in scheme
    if base_name == "weight":
        arg_name = "weights"
    elif base_name == "output":
        arg_name = "output_activations"
    else:
        # (input, q, k, v)
        arg_name = "input_activations"

    quantization_args: Optional[QuantizationArgs] = getattr_chain(
        module, f"quantization_scheme.{arg_name}", None
    )
    if quantization_args is None or quantization_args.is_online():
        return

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
            update_offload_parameter(module, f"{base_name}_global_scale", global_scale)
        else:
            global_scale = getattr(module, f"{base_name}_global_scale", None)

        if should_calculate_qparams:
            updated_scale, updated_zero_point = observer(
                value, g_idx=g_idx, global_scale=global_scale
            )
            # register or update scale & zero_point parameters (supports block shapes)
            scale_name = f"{base_name}_scale"
            zp_name = f"{base_name}_zero_point"
            update_offload_parameter(module, scale_name, updated_scale)
            update_offload_parameter(module, zp_name, updated_zero_point)


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


def calibrate_query_hook(module: Module, query_states: torch.Tensor):
    query_states = query_states.flatten(0, -2)
    calibrate_activations(module, query_states, base_name="q")


def calibrate_key_hook(module: Module, key_states: torch.Tensor):
    key_states = key_states.flatten(0, -2)
    calibrate_activations(module, key_states, base_name="k")


def calibrate_value_hook(module: Module, value_states: torch.Tensor):
    value_states = value_states.flatten(0, -2)
    calibrate_activations(module, value_states, base_name="v")


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

    module.quantization_status = QuantizationStatus.FROZEN


ALL_CALIBRATION_HOOKS = {
    calibrate_input_hook,
    calibrate_query_hook,
    calibrate_key_hook,
    calibrate_value_hook,
    calibrate_output_hook,
}


def reset_quantization_status(model: Module):
    from llmcompressor.modifiers.utils.hooks import HooksMixin

    for module in model.modules():
        # reset status
        if hasattr(module, "quantization_status"):
            delattr(module, "quantization_status")

        # reset observers
        for base_name in ALL_OBSERVER_BASE_NAMES:
            attr_name = f"{base_name}_observer"
            if hasattr(module, attr_name):
                delattr(module, attr_name)

        # remove hooks (note that removal is idempotent)
        for handle_id, hook in module._forward_hooks.items():
            if hook in ALL_CALIBRATION_HOOKS:
                HooksMixin.remove_hooks_by_id(set(handle_id))
