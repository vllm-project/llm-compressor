from collections.abc import Iterable
from itertools import product
from typing import Any

import torch
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationStatus,
)
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from compressed_tensors.utils import (
    getattr_chain,
    update_offload_parameter,
)
from loguru import logger
from torch.nn import Module

from llmcompressor.core import active_session
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
    if args is None or args.dynamic is True:
        return

    observer = args.observer
    if observer is None:
        observer = "memoryless_minmax" if base_name == "weight" else "minmax"

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


def _is_module_collection(value: Any) -> bool:
    """
    Whether a value is a collection of modules rather than a single module.

    Container modules such as `ModuleDict` are both `Module` and `Iterable`, so the
    module check has to come first: iterating a `ModuleDict` yields its keys, which
    are strings, and iterating a one character string yields itself, so recursing
    into a container module never terminates. `str` is excluded for the same reason,
    so that a string passed here by mistake is ignored rather than recursed into.
    """
    return isinstance(value, Iterable) and not isinstance(value, (Module, str))


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
    if _is_module_collection(module):
        for m in module:
            observe(m, base_name)
        return

    observer = getattr(module, f"{base_name}_observer", None)
    if observer is None:
        return

    observer(getattr(module, base_name))


def update_qparams(
    module: Module | Iterable[Module],
    base_name: str | Iterable[str],
    only_update_onload: bool = False,
):
    """
    Compute quantization parameters from observer statistics and store on module.

    For dynamic quantization, scale/zp updates are skipped (scale/zp are
    computed at inference time). For non-TENSOR_GROUP strategies, global_scale
    is None and naturally skipped.

    :param module: torch.nn.Module with attached observer (or iterable of modules)
    :param base_name: substring used to fetch the observer, scales, and zp.
        Can be a string or iterable of strings.
    :only_update_onload: option to only update the onloaded value, useful
        when we want to do a temporary update or in DDP situations where
        we want only want one rank to update the offload+onload to avoid
        multiple writes to the offload (rest just update onload)
    """
    is_collection = _is_module_collection(module)
    if is_collection or not isinstance(base_name, str):
        modules = module if is_collection else [module]
        base_names = [base_name] if isinstance(base_name, str) else base_name
        for m, b in product(modules, base_names):
            update_qparams(m, b, only_update_onload=only_update_onload)
        return

    observer = getattr(module, f"{base_name}_observer", None)
    if observer is None:
        return
    if not observer.has_statistics:
        logger.warning(
            f"Attempted to calibrate a module {base_name} without statistics. This is "
            " often indicative of a bad calibration setup, such as not using "
            "`moe_calibrate_all_experts` or selecting the wrong calibration pipeline."
        )
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
        full_param_name = f"{base_name}_{param_name}"
        if hasattr(module, full_param_name):
            if not only_update_onload:  # update offload + onload
                update_offload_parameter(module, full_param_name, param_val)
            else:  # only update onload
                getattr(module, full_param_name).data = param_val


def _current_loss_mask() -> torch.Tensor | None:
    """
    Loss mask of the batch being calibrated, or None if token masking is off.
    The sequential pipeline fills `state.loss_masks` when `use_loss_mask=True`,
    the same lookup AWQ uses.
    """
    state = active_session().state
    loss_masks = state.loss_masks if state else None
    batch_idx = state.current_batch_idx if state else -1
    if not loss_masks or not 0 <= batch_idx < len(loss_masks):
        return None
    return loss_masks[batch_idx]


def _select_masked_tokens(
    module: Module, value: torch.Tensor, seq_dim: int
) -> torch.Tensor:
    """
    Keep only the tokens where the current batch's loss mask is nonzero, so that
    masked tokens do not affect activation statistics. Without a mask, the value
    is returned unchanged.

    :param module: module being calibrated, named in the error message
    :param value: activations of shape (batch_size, seq_len, hidden_dim) if
        `seq_dim` is 1, or attention states of shape
        (batch_size, num_heads, seq_len, head_dim) if `seq_dim` is 2
    :param seq_dim: dimension of `value` that indexes the tokens
    :return: the kept tokens, with batch size 1 and the dims of `value`
    """
    loss_mask = _current_loss_mask()
    if loss_mask is None:
        return value

    token_shape = tuple(value.shape[:1] + value.shape[seq_dim : seq_dim + 1])
    if value.dim() != seq_dim + 2 or token_shape != tuple(loss_mask.shape):
        raise ValueError(
            f"Token masking (use_loss_mask=True) cannot be applied to "
            f"{type(module).__name__}: its activations have shape "
            f"{tuple(value.shape)}, which does not align with the loss mask of "
            f"shape {tuple(loss_mask.shape)}. This happens for modules which do "
            "not see the batch's token sequence, such as routed MoE experts or "
            "vision encoders. Add the module to `ignore`, or disable token masking."
        )

    keep = loss_mask.to(device=value.device).bool()
    return value.movedim(seq_dim, 1)[keep].unsqueeze(0).movedim(1, seq_dim)


def calibrate_input_hook(module: Module, args: Any):
    """
    Hook to calibrate input activations by accumulating statistics in the observer.
    With `use_loss_mask=True`, only tokens selected by the loss mask are observed.
    """
    args = args[0] if isinstance(args, tuple) else args
    module.input_observer(_select_masked_tokens(module, args, seq_dim=1))


def calibrate_output_hook(module: Module, _args: Any, output: torch.Tensor):
    """
    Hook to calibrate output activations by accumulating statistics in the observer.
    With `use_loss_mask=True`, only tokens selected by the loss mask are observed.
    """
    module.output_observer(_select_masked_tokens(module, output, seq_dim=1))
    output = forward_quantize(
        module=module,
        value=output,
        base_name="output",
        args=module.quantization_scheme.output_activations,
    )
    return output


def calibrate_query_hook(module: Module, query_states: torch.Tensor):
    module.q_observer(_select_masked_tokens(module, query_states, seq_dim=2))


def calibrate_key_hook(module: Module, key_states: torch.Tensor):
    module.k_observer(_select_masked_tokens(module, key_states, seq_dim=2))


def calibrate_value_hook(module: Module, value_states: torch.Tensor):
    module.v_observer(_select_masked_tokens(module, value_states, seq_dim=2))


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

    if module.quantization_status in (
        QuantizationStatus.FROZEN,
        QuantizationStatus.COMPRESSED,
    ):
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
