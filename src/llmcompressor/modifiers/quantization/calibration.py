from typing import Any, Iterable

import torch
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationStatus,
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
    "observe",
    "update_qparams",
    "calibrate_input_hook",
    "calibrate_output_hook",
    "freeze_module_quantization",
    "apply_calibration_status",
    "reset_quantization_status",
    "calibrate_query_hook",
    "calibrate_key_hook",
    "calibrate_value_hook",
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
        observer = Observer.load_from_registry(observer, base_name=base_name, args=args)
        module.register_module(f"{base_name}_observer", observer)
        observer.attach(module)


def observe(
    module: Module | Iterable[Module],
    base_name: str,
):
    """
    Run observers to accumulate statistics on modules.
    Must be called before update_qparams.

    :param module: module or iterable of modules with observer attributes
    :param base_name: substring used to fetch the observer and value to observe
    """
    if isinstance(module, Iterable):
        for m in module:
            observe(m, base_name)
        return

    observer = getattr(module, f"{base_name}_observer", None)
    if observer is None:
        return
    with align_module_device(module):
        observer(getattr(module, base_name))


def update_qparams(
    module: Module | Iterable[Module],
    base_name: str,
):
    """
    Compute quantization parameters from observer statistics and store on module.

    For dynamic quantization, scale/zp updates are skipped (scale/zp are
    computed at inference time). For non-TENSOR_GROUP strategies, global_scale
    is None and naturally skipped.

    :param module: torch.nn.Module with attached observer (or iterable of modules)
    :param base_name: substring used to fetch the observer, scales, and zp
    """
    if isinstance(module, Iterable):
        for m in module:
            update_qparams(m, base_name)
        return

    with align_module_device(module):
        observer = getattr(module, f"{base_name}_observer", None)
        if observer is None:
            return

        # Dynamic (activation) quantization: only store global_scale, not scale/zp
        args = observer.args
        is_dynamic = getattr(args, "dynamic", False) in (True, DynamicType.LOCAL)

        qparams = observer.get_qparams()
        for param_name, param_val in qparams.items():
            if param_val is None:
                continue
            if is_dynamic and param_name in ("scale", "zero_point"):
                continue
            if hasattr(module, f"{base_name}_{param_name}"):
                update_offload_parameter(module, f"{base_name}_{param_name}", param_val)


def calibrate_input_hook(module: Module, args: Any):
    """
    Hook to calibrate input activations by accumulating statistics in the observer.
    """
    args = args[0] if isinstance(args, tuple) else args
    module.input_observer(args)


def calibrate_output_hook(module: Module, _args: Any, output: torch.Tensor):
    """
    Hook to calibrate output activations by accumulating statistics in the observer.
    """
    module.output_observer(output)
    output = forward_quantize(
        module=module,
        value=output,
        base_name="output",
        args=module.quantization_scheme.output_activations,
    )
    return output


def calibrate_query_hook(module: Module, query_states: torch.Tensor):
    module.q_observer(query_states)


def calibrate_key_hook(module: Module, key_states: torch.Tensor):
    module.k_observer(key_states)


def calibrate_value_hook(module: Module, value_states: torch.Tensor):
    module.v_observer(value_states)


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
