from typing import Any, Iterable, Optional, Set, Tuple

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
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from torch.nn import Module

from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales
from llmcompressor.observers import Observer

__all__ = [
    "calibrate_weights",
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
        observer = Observer.load_from_registry(
            observer, base_name=base_name, args=args, module=module
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


def calibrate_weights(
    model: Module,
    *,
    named_modules: Optional[Iterable[Tuple[str, Module]]] = None,
    targets: Optional[Set[str]] = None,
    ignore: Optional[Iterable[str]] = None,
    update_zp_scale: bool = True,
    desc: Optional[str] = "Calibrating weights",
    show_progress: bool = True,
) -> None:
    """
    Traverse the model once (DFS) and run weight calibration: global scales for
    FP4/TENSOR_GROUP, fused layer global scales for Attention/MLP, and weight
    scale/zero-point. Replaces separate loops over named_modules and
    model.modules() for better cache locality and fewer CPU–GPU onloads when
    using offloading.

    Order of operations per module:
    1. Pre-order: update_weight_global_scale for target (quantizable) modules.
    2. Post-order: update_fused_layer_weight_global_scales for every module
       (no-op except for Attention/MLP containers); then update_weight_zp_scale
       for target modules if update_zp_scale is True.

    :param model: Root module to traverse (e.g. state.model).
    :param named_modules: Optional list of (name, module) for target modules.
        If provided, only these modules get global_scale and zp_scale; enables
        DDP by passing this rank's subset (see #2220). If None, targets and
        ignore must be provided and match_named_modules(model, targets, ignore)
        is used.
    :param targets: Target module name patterns (used when named_modules is None).
    :param ignore: Ignore patterns (used when named_modules is None).
    :param update_zp_scale: If True, call update_weight_zp_scale on target
        modules in post-order. Set False for modifiers that do zp_scale in
        hooks (e.g. GPTQ).
    :param desc: Progress bar description; None to disable progress bar.
    :param show_progress: If True and desc is not None, show a tqdm progress bar.
    """
    if named_modules is None:
        if targets is None or ignore is None:
            raise ValueError(
                "calibrate_weights requires either named_modules or both "
                "targets and ignore"
            )
        named_modules = list(match_named_modules(model, targets, ignore))
    else:
        named_modules = list(named_modules)

    target_set = {id(m) for _, m in named_modules}
    total_targets = len(target_set)

    try:
        import tqdm
    except ImportError:
        tqdm = None

    if show_progress and desc is not None and tqdm is not None and total_targets > 0:
        pbar = tqdm.tqdm(total=total_targets, desc=desc)
    else:
        pbar = None

    # Stack-based DFS: (module, children_visited)
    stack: list[Tuple[Module, bool]] = [(model, False)]

    while stack:
        module, children_done = stack.pop()

        if not children_done:
            # Pre-order: global scale for target modules (FP4 / TENSOR_GROUP)
            if id(module) in target_set:
                update_weight_global_scale(module)
            stack.append((module, True))
            for child in reversed(list(module.children())):
                stack.append((child, False))
        else:
            # Post-order: fused global scales (Attention/MLP), then zp_scale for targets
            update_fused_layer_weight_global_scales(module)
            if update_zp_scale and id(module) in target_set:
                update_weight_zp_scale(module)
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()


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
            delattr(module, obs_name)

    module.quantization_status = QuantizationStatus.FROZEN


def reset_quantization_status(model: Module):
    for module in model.modules():
        if hasattr(module, "quantization_status"):
            delattr(module, "quantization_status")
