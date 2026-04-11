from typing import Any, Optional

import torch
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationStatus,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.utils import (
    align_module_device,
    getattr_chain,
    update_offload_parameter,
)
from loguru import logger
from torch.nn import Module

from llmcompressor.observers import Observer

__all__ = [
    "initialize_observer",
    "call_observer",
    "calibrate_input_hook",
    "calibrate_output_hook",
    "freeze_module_quantization",
    "apply_calibration_status",
    "reset_quantization_status",
    "calibrate_query_hook",
    "calibrate_key_hook",
    "calibrate_value_hook",
    "has_quantization_weights",
    "is_tensor_group_weight",
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

    This function always initializes memoryless observers for weights

    :param module: torch.nn.Module that the observer is being attached to
    :param base_name: str used to name the observer attribute

    """
    if base_name == "weight":
        arg_name = "weights"
    elif base_name == "output":
        arg_name = "output_activations"
    else:  # input, q, k, v
        arg_name = "input_activations"

    args: QuantizationArgs = getattr_chain(
        module, f"quantization_scheme.{arg_name}", None
    )
    observer = args.observer

    # training is no longer supported: always use memoryless for weights
    if base_name == "weight" and args.observer in ("static_minmax", "minmax"):
        observer = "memoryless_minmax"
        logger.warning(
            "Overriding weight observer for lower memory usage "
            f"({args.observer} -> {observer})",
            log_once=True,
        )
    if base_name == "weight" and args.observer in ("mse",):
        observer = "memoryless_mse"
        logger.warning(
            "Overriding weight observer for lower memory usage "
            f"({args.observer} -> {observer})",
            log_once=True,
        )

    if args is not None and args.dynamic is not True:
        observer = Observer.load_from_registry(
            observer, base_name=base_name, args=args, module=module
        )
        module.register_module(f"{base_name}_observer", observer)
        observer.attach(module)


def call_observer(
    module: Module,
    base_name: str,
    value: Optional[torch.Tensor] = None,
    should_calculate_gparam: bool = False,  # Kept for backward compatibility, ignored
    should_calculate_qparams: bool = True,  # Kept for backward compatibility, ignored
):
    """
    Call a module's attached input/weight/output observer using a provided value.
    Update the module's scale, zero_point, and global_scale using the observer's return values.

    The observer's get_qparams() method automatically computes global_scale for TENSOR_GROUP strategy.

    :param module: torch.nn.Module
    :param base_name: substring used to fetch the observer, scales, and zp
    :param value: torch.Tensor to be passed to the observer for activations. If
        base_name is "weight", then the module's weight tensor will be used
    :param should_calculate_gparam: Kept for backward compatibility, ignored
    :param should_calculate_qparams: Kept for backward compatibility, ignored
    """
    with align_module_device(module):
        if value is None and base_name == "weight":
            value = module.weight
        observer: Observer = getattr(module, f"{base_name}_observer")

        qparams = observer(value).get_qparams()

        for param_name, param_val in qparams.items():
            if param_val is not None:
                update_offload_parameter(
                    module, f"{base_name}_{param_name}", param_val
                )


def has_quantization_weights(module: Module) -> bool:
    """
    Check if a module has weight quantization configured.

    :param module: module to check
    :return: True if module has quantization_scheme.weights
    """
    return getattr_chain(module, "quantization_scheme.weights", None) is not None


def is_tensor_group_weight(module: Module) -> bool:
    """
    Check if a module uses TENSOR_GROUP strategy for weight quantization.

    :param module: module to check
    :return: True if module uses TENSOR_GROUP strategy for weights
    """
    return (
        getattr_chain(module, "quantization_scheme.weights.strategy", None)
        == QuantizationStrategy.TENSOR_GROUP
    )


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

    field_name = "input" if base_name != "output" else "output"  # input,q,k,v,output
    args_attr = f"quantization_scheme.{field_name}_activations"
    quantization_args = getattr_chain(module, args_attr, None)

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


def calibrate_query_hook(module: Module, query_states: torch.Tensor):
    calibrate_activations(module, query_states, base_name="q")


def calibrate_key_hook(module: Module, key_states: torch.Tensor):
    calibrate_activations(module, key_states, base_name="k")


def calibrate_value_hook(module: Module, value_states: torch.Tensor):
    calibrate_activations(module, value_states, base_name="v")


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
    for name in ("input", "weight", "output", "q", "k", "v"):
        obs_name = f"{name}_observer"
        if hasattr(module, obs_name):
            getattr(module, obs_name).detach(module)
            delattr(module, obs_name)

    module.quantization_status = QuantizationStatus.FROZEN


def reset_quantization_status(model: Module):
    for module in model.modules():
        if hasattr(module, "quantization_status"):
            delattr(module, "quantization_status")
