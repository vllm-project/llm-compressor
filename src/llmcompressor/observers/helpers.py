"""
Helper functions for observer token counting and analysis.

Provides utility functions for analyzing observer statistics
and token counts across model modules. Used for monitoring compression
effects and understanding model behavior during quantization and
pruning operations.
"""

from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.utils import strategy_cdiv

__all__ = ["flatten_for_calibration"]


def flatten_for_calibration(
    value: torch.Tensor,
    base_name: str,
    args: QuantizationArgs,
    g_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reshapes the value according to the quantization strategy for the purposes of
    scale/zp calibration. The value after flattening has the following shape:

    `(num_observations, *qparam_shape, group_size)`

    The first dim is the number of observations (usually the batch size times number of
    tokens), the middle dims are the dimension of the scales, and the last dim is the
    number of elements being quantized per group.

    :param value: value being flattened
    :param base_name: weight, input, output, q/k/v. Used to characterize the value as
        being a weight, activation, or attention state
    :param args: quantization args for determining how the value is flattened
    :param g_idx: optional gidx for weight activation ordering
    :return: value which has been reshaped for calibration
    """
    if base_name == "weight":
        return _flatten_weight(value, args, g_idx)
    elif base_name in ("input", "output"):
        return _flatten_activation(value, args)
    elif base_name in ("q", "k", "v"):
        return _flatten_attention(value, args)
    else:
        raise ValueError(f"Unknown quantization base name: {base_name}")


def _flatten_weight(
    value: torch.Tensor, args: QuantizationArgs, g_idx: Optional[torch.Tensor] = None
):
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
        if g_idx is not None:
            value = value.index_select(dim=1, index=torch.argsort(g_idx))

        # (1, num_rows, num_groups, group_size)
        return value.unflatten(-1, (-1, args.group_size)).unsqueeze(0)

    if args.strategy == QuantizationStrategy.BLOCK:
        # (1, num_block_rows, num_block_cols, block_width * block_height)
        block_height, block_width = args.block_structure
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

    assert False, f"Unknown strategy {args.strategy}"


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

    assert False, f"Unknown strategy {args.strategy}"


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

    assert False, f"Unknown strategy {args.strategy}"
