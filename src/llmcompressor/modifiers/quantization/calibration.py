import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Iterator, Optional, Tuple

import torch
import tqdm
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


def _post_order_modules(model: Module) -> Iterator[Module]:
    """Yield every module in the tree in DFS post-order."""
    stack: list[Tuple[Module, bool]] = [(model, False)]
    while stack:
        module, children_done = stack.pop()
        if not children_done:
            stack.append((module, True))
            for child in reversed(list(module.children())):
                stack.append((child, False))
        else:
            yield module


def _update_weight_calibration_once(module: Module, update_zp_scale: bool) -> None:
    """
    Onload weight once and run both global scale (gparam) and scale/zp (qparams).
    Used in sequential DFS to avoid double onload for NVFP4.
    """
    if getattr_chain(module, "quantization_scheme.weights", None) is None:
        return
    need_gparam = (
        getattr_chain(module, "quantization_scheme.weights.strategy", None)
        == QuantizationStrategy.TENSOR_GROUP
    )
    need_qparams = update_zp_scale
    if not need_gparam and not need_qparams:
        return
    if (
        need_qparams
        and getattr(module, "quantization_status", None)
        != QuantizationStatus.CALIBRATION
    ):
        logger.warning(
            "Attempting to calibrate weights of a module not in calibration mode"
        )
    with align_module_device(module):
        value = module.weight
        call_observer(
            module,
            base_name="weight",
            value=value,
            should_calculate_gparam=need_gparam,
            should_calculate_qparams=need_qparams,
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
    targets: Iterable[str] = (),
    ignore: Iterable[str] = (),
    update_zp_scale: bool = True,
    desc: Optional[str] = "Calibrating weights",
    show_progress: bool = True,
    parallel: bool = False,
    max_workers: Optional[int] = None,
) -> None:
    """
    Run weight calibration: per-tensor global scale (gparam), fused global scales
    for Attention/MLP, and scale/zero-point (qparams). Minimizes weight onloads
    when using offloading (one onload per target in the default path).

    Two modes:
    - Sequential (parallel=False): DFS over the model. Pre-order: one onload per
      target via _update_weight_calibration_once (gparam + qparams). Post-order:
      update_fused_layer_weight_global_scales (no extra onload for targets).
    - Parallel (parallel=True): Phase 1 runs gparam + qparams per target
      (order-independent, parallelizable). Phase 2 applies fused global scales
      and rescales per-tensor scale s' = s * (g' / g).

    DDP: Works with distributed setups. Pass named_modules as this rank's
    subset so each rank only calibrates its assigned modules (see e.g. #2220).
    Activation observer sync across ranks is handled by
    QuantizationMixin.sync_activation_observers at layer
    boundaries (PR #2391); weight calibration does not all-reduce weight
    observer state—each rank calibrates its subset and can broadcast
    quantized params afterward (e.g. GPTQ-style) if needed. Fused groups
    (q/k/v, gate/up) must be assigned to the same rank so
    update_fused_layer_weight_global_scales sees the full group. For
    balanced wall time, assign by weight size (e.g. greedy_bin_packing with
    item_weight_fn=lambda m: m.weight.numel(); see GPTQ DDP #2333 which uses
    hessian shape for the same idea).

    Benchmark: See tests/benchmark_calibrate_weights.py for onload count and
    single-vs-double-onload timing.

    :param model: Root module to traverse (e.g. state.model).
    :param named_modules: If provided, only these (name, module) pairs are
        calibrated; enables DDP by passing this rank's subset. If None, uses
        match_named_modules(model, targets, ignore).
    :param targets: Name patterns when named_modules is None. Default ().
    :param ignore: Ignore patterns when named_modules is None. Default ().
    :param update_zp_scale: If True, compute scale/zp for targets. False for
        modifiers that do zp in hooks (e.g. GPTQ).
    :param desc: Progress bar description; None disables bar.
    :param show_progress: If True and desc set, show tqdm bar.
    :param parallel: If True, use two-phase parallel calibration.
    :param max_workers: If parallel and int, phase 1 uses this many workers.
    """
    if named_modules is None:
        named_modules = list(match_named_modules(model, targets, ignore))
    else:
        named_modules = list(named_modules)
    # DDP: target_set = only these get gparam + qparams (this rank's subset).
    target_set = {m for _, m in named_modules}
    target_list = list(target_set)
    total_targets = len(target_list)

    if show_progress and desc is not None and total_targets > 0:
        pbar = tqdm.tqdm(total=total_targets, desc=desc)
    else:
        pbar = None

    if parallel:
        # Phase 1: per-module global scale + scale/zp (order-independent)
        pbar_lock = threading.Lock()

        def _phase1_one(module: Module) -> None:
            update_weight_global_scale(module)
            if update_zp_scale:
                update_weight_zp_scale(module)
            if pbar is not None:
                with pbar_lock:
                    pbar.update(1)

        if max_workers is not None and max_workers > 0:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(_phase1_one, target_list))
        else:
            for module in target_list:
                _phase1_one(module)

        # Phase 2: fused global scales (rescale per-tensor scale s' = s * g' / g)
        for module in _post_order_modules(model):
            update_fused_layer_weight_global_scales(module)
    else:
        # Sequential DFS: pre-order one onload for gparam + qparams, post-order fused
        seen_pre: set[Module] = set()
        seen_post: set[Module] = set()
        stack = [(model, False)]
        while stack:
            module, children_done = stack.pop()
            if not children_done:
                if module in target_set and module not in seen_pre:
                    seen_pre.add(module)
                    _update_weight_calibration_once(module, update_zp_scale)
                stack.append((module, True))
                for child in reversed(list(module.children())):
                    stack.append((child, False))
            else:
                update_fused_layer_weight_global_scales(module)
                if update_zp_scale and module in target_set and module not in seen_post:
                    seen_post.add(module)
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
