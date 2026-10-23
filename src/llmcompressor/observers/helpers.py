"""
Helper functions for observer token counting and analysis.

Provides utility functions for analyzing observer statistics
and token counts across model modules. Used for monitoring compression
effects and understanding model behavior during quantization and
pruning operations.
"""

from collections.abc import Collection

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.utils import (
    maybe_pad_tensor_for_block_quant,
    strategy_cdiv,
)
from compressed_tensors.utils import getattr_chain
from torch.nn import Module

from llmcompressor.observers.fused_mappings import get_fused_layers

__all__ = [
    "flatten_for_calibration",
    "fuse_weight_observers",
    "validate_fused_layer_scope",
    "ACTIVATION_OBS",
]

ACTIVATION_OBS = ("input", "output", "q", "k", "v")


def flatten_for_calibration(
    value: torch.Tensor,
    base_name: str,
    args: QuantizationArgs,
) -> torch.Tensor:
    """
    Reshapes the value according to the quantization strategy for the purposes of
    scale/zp calibration. The value after flattening has the following shape:

    `(num_observations, *qparam_shape, group_size)`

    For block quantization, value will be zero-padded if it is not evenly
    divisible by block_size, so as not to distort the calculated qparams and to be
    compatible with vllm block-wise kernels that do not require even divisibility.

    The first dim is the number of observations (usually the batch size times number of
    tokens), the middle dims are the dimension of the scales, and the last dim is the
    number of elements being quantized per group.

    :param value: value being flattened
    :param base_name: weight, input, output, q/k/v. Used to characterize the value as
        being a weight, activation, or attention state
    :param args: quantization args for determining how the value is flattened
    :return: value which has been reshaped for calibration
    """
    if base_name == "weight":
        return _flatten_weight(value, args)
    elif base_name in ("input", "output"):
        return _flatten_activation(value, args)
    elif base_name in ("q", "k", "v"):
        return _flatten_attention(value, args)
    else:
        raise ValueError(f"Unknown quantization base name: {base_name}")


def _flatten_weight(value: torch.Tensor, args: QuantizationArgs):
    # value.shape = (num_rows, num_cols)

    if args.strategy == QuantizationStrategy.TENSOR:
        # (1, 1, num_weight_elems)
        return value.reshape((1, 1, -1))

    if args.strategy == QuantizationStrategy.TOKEN:
        raise ValueError("Token quantization cannot be applied to weights")

    if args.strategy == QuantizationStrategy.CHANNEL:
        # (1, num_rows, 1, num_cols)
        return value.unsqueeze(-2).unsqueeze(0)

    if args.strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        # (1, num_rows, num_groups, group_size)
        return value.unflatten(-1, (-1, args.group_size)).unsqueeze(0)

    if args.strategy == QuantizationStrategy.BLOCK:
        # (1, num_block_rows, num_block_cols, block_width * block_height)
        block_height, block_width = args.block_structure

        # If shape is incompatible, zero-pad to compatible shape
        # so as not to distort resultant qparams
        value = maybe_pad_tensor_for_block_quant(value, args.block_structure)
        rows, cols = value.shape

        block_rows = strategy_cdiv(rows, block_height, args.strategy, strict=True)
        block_cols = strategy_cdiv(cols, block_width, args.strategy, strict=True)
        return (
            value.reshape(block_rows, block_height, block_cols, block_width)
            .transpose(1, 2)
            .flatten(-2, -1)
            .unsqueeze(0)
        )

    if args.strategy == QuantizationStrategy.ATTN_HEAD:
        raise ValueError("Attention head quantization cannot be applied to weights")

    raise ValueError(f"Unknown strategy {args.strategy}")


def _flatten_activation(value: torch.Tensor, args: QuantizationArgs):
    # value.shape = (batch_size, seq_len, hidden_dim)

    if args.strategy == QuantizationStrategy.TENSOR:
        # (batch_size * seq_len, 1, hidden_dim)
        return value.reshape((-1, 1, value.size(-1)))

    if args.strategy == QuantizationStrategy.TOKEN:
        # (batch_size, seq_len, hidden_dim)
        # warning: token quantization uses `compute_dynamic_scales_and_zp`
        return value

    if args.strategy == QuantizationStrategy.CHANNEL:
        raise ValueError("Channel quantization cannot be applied to activations")

    if args.strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        # (batch_size * seq_len, num_groups, group_size)
        # warning: group activation quantization uses compute_dynamic_scales_and_zp
        return value.flatten(0, 1).unflatten(-1, (-1, args.group_size))

    if args.strategy == QuantizationStrategy.BLOCK:
        raise ValueError("Block quantization cannot be applied to activations")

    if args.strategy == QuantizationStrategy.ATTN_HEAD:
        raise ValueError("Attention head quantization cannot be applied to activations")

    raise ValueError(f"Unknown strategy {args.strategy}")


def _flatten_attention(value: torch.Tensor, args: QuantizationArgs):
    # value.shape = (batch_size, num_heads, seq_len, head_dim)

    if args.strategy == QuantizationStrategy.TENSOR:
        # (batch_size * seq_len, 1, num_heads * head_dim)
        return value.transpose(1, 2).flatten(0, 1).flatten(-2, -1).unsqueeze(-2)

    if args.strategy == QuantizationStrategy.TOKEN:
        raise ValueError("Token quantization cannot be applied to attention")

    if args.strategy == QuantizationStrategy.CHANNEL:
        raise ValueError("Channel quantization cannot be applied to attention")

    if args.strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        raise ValueError("Group quantization cannot be applied to attention")

    if args.strategy == QuantizationStrategy.BLOCK:
        raise ValueError("Block quantization cannot be applied to attention")

    if args.strategy == QuantizationStrategy.ATTN_HEAD:
        # (batch_size * seq_len, num_heads, 1, 1, head_dim)
        return value.transpose(1, 2).flatten(0, 1).unsqueeze(-2).unsqueeze(-2)

    raise ValueError(f"Unknown strategy {args.strategy}")


def _fused_layers_in_scope(
    fused_layers: dict[str, Module], module_scope: set[Module]
) -> bool:
    group = set(fused_layers.values())
    if group.isdisjoint(module_scope):
        return False
    if group.issubset(module_scope):
        return True

    if any(
        layer in module_scope
        and (
            getattr_chain(layer, "quantization_scheme.weights.strategy", None)
            == QuantizationStrategy.TENSOR_GROUP
            or getattr_chain(layer, "weight_observer.args.strategy", None)
            == QuantizationStrategy.TENSOR_GROUP
        )
        for layer in fused_layers.values()
    ):
        raise ValueError(
            f"TENSOR_GROUP fused layers {tuple(fused_layers)} span multiple sequential "
            "calibration subgraphs. Choose `sequential_targets` so all fused layers "
            "are included in one subgraph (for example, target their common decoder "
            "or expert module instead of individual Linear modules)."
        )
    return False


def validate_fused_layer_scope(model: Module, modules: Collection[Module]):
    """Validate TENSOR_GROUP scopes, including modifiers that do not use observers."""
    module_scope = set(modules)
    for submodule in model.modules():
        for fused_layers in get_fused_layers(submodule):
            _fused_layers_in_scope(fused_layers, module_scope)


def fuse_weight_observers(model: Module, modules: Collection[Module] | None = None):
    """
    Link weight observers across fused layer groups for shared global_scale.

    For TENSOR_GROUP quantization (e.g. NVFP4), vLLM requires that fused
    layers (Q/K/V attention, gate/up MLP) share the same global_scale.
    This function links their observers so that get_qparams() computes
    global_scale from the combined statistics of all observers in the group.
    Fused groups are defined in llmcompressor.observers.fused_mappings.

    :param model: model whose weight observers should be linked
    :param modules: optional active calibration scope. Only groups fully in scope are
        fused; a partially scoped TENSOR_GROUP group raises before calibration starts.
    """
    from llmcompressor.observers.fusion import FusionHandler

    module_scope = None if modules is None else set(modules)
    for submodule in model.modules():
        for fused_layers in get_fused_layers(submodule):
            layers_to_fuse = list(fused_layers.values())
            fusion_name_group = tuple(fused_layers.keys())

            if module_scope is not None and not _fused_layers_in_scope(
                fused_layers, module_scope
            ):
                continue

            only_obs = True
            only_tensor_group = True
            observers_and_modules = []
            for layer in layers_to_fuse:
                obs = getattr(layer, "weight_observer", None)
                if obs is None:
                    only_obs = False
                    continue
                if obs.args.strategy != QuantizationStrategy.TENSOR_GROUP:
                    only_tensor_group = False
                    continue
                observers_and_modules.append((obs, layer))
            if len(observers_and_modules) == 0:
                continue

            start = f"Some layers in fused group: {fusion_name_group} have"
            end = ", need all fused layers to have same quantization"
            assert only_obs, f"{start} no weight observer{end}"
            assert only_tensor_group, f"{start} non-TENSOR_GROUP quantization{end}"

            FusionHandler.fuse(observers_and_modules)


def lerp(start: torch.Tensor, end: torch.Tensor, weight: float) -> torch.Tensor:
    """Linear interpolation — torch.lerp is not implemented for all dtypes."""
    return (start * (1.0 - weight)) + (end * weight)
