from typing import Any, Iterable, Optional

import torch
from compressed_tensors.distributed import update_module_parallel
from compressed_tensors.offload import is_distributed
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStatus,
    QuantizationStrategy,
    quant_metadata,
)
from compressed_tensors.utils import (
    align_module_device,
    deprecated,
    getattr_chain,
    update_offload_parameter,
)
from loguru import logger
from torch.nn import Module
from tqdm import tqdm

from llmcompressor.observers import Observer
from llmcompressor.utils.dist import wait_for_comms

__all__ = [
    "initialize_observer",
    "update_weight_zp_scale",
    "calibrate_input_hook",
    "calibrate_output_hook",
    "freeze_module_quantization",
    "apply_calibration_status",
    "reset_quantization_status",
    "update_weight_global_scale",
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


@deprecated("Observer.calculate_qparams")
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
        if value is None and base_name == "weight":
            value = module.weight
        observer: Observer = getattr(module, f"{base_name}_observer")

        if should_calculate_gparam:
            global_scale = observer.get_global_scale(value)
            update_offload_parameter(module, f"{base_name}_global_scale", global_scale)

        if should_calculate_qparams:
            scale, zero_point = observer(value)
            update_offload_parameter(module, f"{base_name}_scale", scale)
            if hasattr(module, f"{base_name}_zero_point"):
                update_offload_parameter(module, f"{base_name}_zero_point", zero_point)


@deprecated("update_weight_qparams")
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


def update_weight_qparams(
    modules: Iterable[torch.nn.Module], show_progress: bool = True
):
    modules: set[torch.nn.Module] = set(
        module for module in modules if hasattr(module, "weight_observer")
    )

    def apply_fn(module: torch.nn.Module):
        observer = module.weight_observer
        observer(module.weight)
        observer.calibrate_module(module)

    desc = "Calculating weight quantization parameters" if show_progress else None
    if not is_distributed():
        for module in tqdm(modules, desc=desc, disable=(not show_progress)):
            apply_fn(module)
    else:
        update_names = quant_metadata.QuantizationMetadata.all_qparam_names()
        update_module_parallel(list(modules), apply_fn, update_names, desc=desc)


def update_activation_qparams(
    modules: Iterable[torch.nn.Module], show_progress: bool = True
):
    modules: set[torch.nn.Module] = set(
        module for module in modules if hasattr(module, "input_observer")
    )

    def apply_fn(module: torch.nn.Module):
        observer = module.input_observer
        observer.calibrate_module(module)

    desc = "Calculating activation quantization parameters"
    if not is_distributed():
        for module in tqdm(modules, desc=desc, disable=(not show_progress)):
            apply_fn(module)
    else:
        import torch.distributed as dist
        from compressed_tensors.distributed.assign import greedy_bin_packing

        _, _, assigned_rank = greedy_bin_packing(
            modules, dist.get_world_size(), weight_fn
        )

        sync = []
        for module in modules:
            print(module.input_observer)
            sync.extend(module.input_observer.synchronize())
        wait_for_comms(sync)

        # update_names = quant_metadata.QuantizationMetadata.all_qparam_names()
        # update_module_parallel(list(modules), apply_fn, update_names, desc=desc)


def calibrate_input_hook(module: Module, args: Any):
    input = args[0] if isinstance(args, tuple) else args
    module.input_observer(input)


def calibrate_output_hook(module: Module, _args: Any, output: torch.Tensor):
    module.output_observer(output)


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
            delattr(module, obs_name)

    module.quantization_status = QuantizationStatus.FROZEN


def reset_quantization_status(model: Module):
    for module in model.modules():
        if hasattr(module, "quantization_status"):
            delattr(module, "quantization_status")
