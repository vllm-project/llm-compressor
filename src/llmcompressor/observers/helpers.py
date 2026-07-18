"""
Helper functions for observer token counting and analysis.

Provides utility functions for analyzing observer statistics
and token counts across model modules. Used for monitoring compression
effects and understanding model behavior during quantization and
pruning operations.
"""

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.utils import (
    maybe_pad_tensor_for_block_quant,
    strategy_cdiv,
)
from torch.nn import Module

__all__ = [
    "flatten_for_calibration",
    "fuse_weight_observers",
    "FUSED_LAYER_NAMES",
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


# Defines which layer names should have their global_scale fused together.
# These sets are used for TENSOR_GROUP quantization (e.g., NVFP4).
FUSED_LAYER_NAMES = [
    # MLP / expert layers have fused gate_up_proj
    ("gate_proj", "up_proj"),
    # Attention layers have fused qkv_proj
    ("q_proj", "k_proj", "v_proj"),
    # DeepSeek multi-latent attention has fused_qkv_a_proj
    ("q_a_proj", "kv_a_proj_with_mqa"),
    # MoE expert layers may use w1/w3 naming
    ("w1", "w3"),
]


def fuse_weight_observers(model: Module):
    """
    Link weight observers across fused layer groups for shared global_scale.

    For TENSOR_GROUP quantization (e.g. NVFP4), vLLM requires that fused
    layers (Q/K/V attention, gate/up MLP) share the same global_scale.
    This function links their observers so that get_qparams() computes
    global_scale from the combined statistics of all observers in the group.

    :param model: model whose weight observers should be linked
    """
    from llmcompressor.observers.fusion import FusionHandler

    for submodule_name, submodule in model.named_modules():
        for fusion_name_group in FUSED_LAYER_NAMES:
            if not all(hasattr(submodule, name) for name in fusion_name_group):
                continue
            layers_to_fuse = [getattr(submodule, name) for name in fusion_name_group]

            only_obs = True
            only_tensor_group = True
            observers_and_modules = []
            for layer in layers_to_fuse:
                if layer is None:
                    continue
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

            start = f"Some layers in fused group: {submodule_name} {fusion_name_group} "
            end = ", need all fused layers to have same quantization"
            if not only_obs:
                raise ValueError(f"{start} have no weight observer{end}")
            if not only_tensor_group:
                raise ValueError(f"{start} have non-TENSOR_GROUP quantization{end}")

            FusionHandler.fuse(observers_and_modules)


def lerp(start: torch.Tensor, end: torch.Tensor, weight: float) -> torch.Tensor:
    """Linear interpolation — torch.lerp is not implemented for all dtypes."""
    return (start * (1.0 - weight)) + (end * weight)
