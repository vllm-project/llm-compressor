"""
Utility functions for handling non-divisible tensor dimensions in block quantization.

When using FP8 block quantization, tensor dimensions must be divisible by the block
size (typically 128x128). For models with non-divisible dimensions (e.g., DeepSeek-V2
with intermediate_size=10944), we need to pad weights during quantization/saving so
that inference frameworks like vLLM can handle them correctly.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "pad_weight_for_block_quant",
    "calculate_block_padding",
    "needs_block_padding",
]


def calculate_block_padding(
    shape: Tuple[int, int],
    block_structure: List[int],
) -> Tuple[int, int]:
    """
    Calculate padding needed to make tensor dimensions divisible by block size.

    :param shape: Original tensor shape (out_features, in_features)
    :param block_structure: Block size as [block_n, block_k]
    :return: Tuple of (pad_n, pad_k) representing padding for each dimension
    """
    block_n, block_k = block_structure
    out_features, in_features = shape

    pad_n = (block_n - out_features % block_n) % block_n
    pad_k = (block_k - in_features % block_k) % block_k

    return pad_n, pad_k


def needs_block_padding(
    shape: Tuple[int, int],
    block_structure: List[int],
) -> bool:
    """
    Check if a tensor with the given shape needs padding for block quantization.

    :param shape: Original tensor shape (out_features, in_features)
    :param block_structure: Block size as [block_n, block_k]
    :return: True if padding is needed, False otherwise
    """
    pad_n, pad_k = calculate_block_padding(shape, block_structure)
    return pad_n > 0 or pad_k > 0


def pad_weight_for_block_quant(
    weight: torch.Tensor,
    block_structure: List[int],
) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
    """
    Pad weight tensor so dimensions are divisible by block size.

    For FP8 block quantization, dimensions must be divisible by the block size
    (typically 128). This function pads the weight tensor with zeros if needed.

    :param weight: Weight tensor of shape (out_features, in_features)
    :param block_structure: Block size as [block_n, block_k]
    :return: Tuple of (padded_weight, original_shape) where original_shape is
             the shape before padding, or None if no padding was applied
    """
    if weight.ndim != 2:
        raise ValueError(
            f"Expected 2D weight tensor, got shape {weight.shape}. "
            "Block quantization padding only supports 2D weight matrices."
        )

    out_features, in_features = weight.shape
    original_shape = (out_features, in_features)

    pad_n, pad_k = calculate_block_padding(original_shape, block_structure)

    if pad_n > 0 or pad_k > 0:
        # F.pad uses (left, right, top, bottom) for 2D tensors
        # For weight (out, in), we pad (0, pad_k, 0, pad_n)
        # This pads the input dimension (columns) by pad_k and
        # the output dimension (rows) by pad_n
        padded_weight = F.pad(weight, (0, pad_k, 0, pad_n), mode="constant", value=0)
        return padded_weight, original_shape

    return weight, None
