from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist
from compressed_tensors.offload import is_distributed, OffloadCache, disable_onloading, as_broadcastable
from compressed_tensors.offload.cache import DeviceCache
from compressed_tensors.offload.utils import module_size
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


# TODO: move
def finalize_distributed_update(modules: Iterable[torch.nn.Module], assigned_rank, update_param_names: Iterable[str]):
    def is_fsdp(module):
        return (
            isinstance(module._parameters, DeviceCache)
            and module._parameters.onload_device != module._parameters.offload_device
        )
    
    def is_same_device_offload(module):
        return (
            isinstance(module._parameters, DeviceCache)
            and module._parameters.onload_device == module._parameters.offload_device
        )


    for module in modules:
        src = assigned_rank[module]

        # # fsdp device offloads
        # if is_fsdp(module):
        #     with disable_onloading():
        #         for name in update_param_names:
        #             if hasattr(module, name):
        #                 dist.broadcast(as_broadcastable(getattr(module, name)), src=src)

        # device onloads
        for name in update_param_names:
            if True: #if OffloadCache.offloading_disabled or is_same_device_offload(module):
                value = getattr(module, name, None)
                if value is not None:
                    logger.info(f"Rank {dist.get_rank()}: {name} shape={value.shape}, "
                        f"dtype={value.dtype}, device={value.device}, "
                        f"is_contiguous={value.is_contiguous()}, "
                        f"stride={value.stride()}, src={src}")
                    dist.broadcast(as_broadcastable(value), src=src)  # doesn't work
                    logger.info(f"Finished {name}")
                    #dist.barrier()  # WORKS


def update_weight_qparams(
    modules: Iterable[torch.nn.Module], show_progress: bool = True
):
    modules: set[torch.nn.Module] = list(set(
        module for module in modules if hasattr(module, "weight_observer")
    ))

    desc = "Calculating weight quantization parameters" if show_progress else None
    if not is_distributed():
        for module in tqdm(modules, desc=desc, disable=(not show_progress)):
            observer = module.weight_observer
            observer(module.weight)
            observer.calibrate_module(module)
    else:
        import torch.distributed as dist
        from compressed_tensors.distributed.assign import greedy_bin_packing

        _, _, assigned_rank = greedy_bin_packing(
            list(modules), dist.get_world_size(), module_size
        )
        
        for module in modules:
            if assigned_rank[module] == dist.get_rank():
                observer = module.weight_observer
                observer(module.weight)
                observer.calibrate_module(module)

        update_names = quant_metadata.QuantizationMetadata.all_qparam_names()
        finalize_distributed_update(modules, assigned_rank, update_names)


def update_activation_qparams(
    modules: Iterable[torch.nn.Module], show_progress: bool = True
):
    modules: set[torch.nn.Module] = list(set(
        module for module in modules if hasattr(module, "input_observer")
    ))

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
            list(modules), dist.get_world_size(), module_size
        )

        comms = []
        for module in modules:
            observer: Observer = module.input_observer
            comms.extend(observer.synchronize(assigned_rank[module]))
        wait_for_comms(comms)
        
        # for module in modules:
        #     if assigned_rank[module] == dist.get_rank():
        #         observer: Observer = module.input_observer
        #         observer.calibrate_module(module)

        # update_names = quant_metadata.QuantizationMetadata.all_qparam_names()
        # finalize_distributed_update(modules, assigned_rank, update_names)


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
