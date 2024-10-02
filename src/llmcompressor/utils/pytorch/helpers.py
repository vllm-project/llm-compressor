import gc
from typing import Any, Optional, Tuple

import torch
from torch import Tensor


@torch.no_grad()
def mse_loss_with_chunking(
    tensor_a: Tensor,
    tensor_b: Tensor,
    device: torch.device,
    max_chunk_memory: int = 1024 * 1024 * 1024,
):
    """
    Calculate the MSE loss between the two tensors
    with chunking to save memory

    :param tensor_a: The first tensor
    :param tensor_b: The second tensor
    :param device: device to use for computation
    :param max_chunk_memory: maximum chunk memory to use
    :return: MSE loss between the two tensors
    """
    loss = 0.0
    fp16_output_flat = tensor_a.view(-1)
    int_w_output_flat = tensor_b.view(-1)
    num_elements = fp16_output_flat.size(0)
    element_size_bytes = tensor_a.element_size()

    # Calculate chunk size dynamically based on max_chunk_memory
    # Divide the max_chunk_memory by twice the element size
    chunk_size = max_chunk_memory // (element_size_bytes * 2)
    chunk_size = min(chunk_size, num_elements)

    # Split the computation into chunks
    tensor_a_chunks = torch.split(fp16_output_flat, chunk_size)
    tensor_b_chunks = torch.split(int_w_output_flat, chunk_size)

    # Compute the loss for each chunk
    for chunk_a, chunk_b in zip(tensor_a_chunks, tensor_b_chunks):
        chunk_loss = (
            (chunk_a.to(device) - chunk_b.to(device)).float().pow(2).sum().item()
        )
        loss += chunk_loss

    loss /= num_elements

    return loss


def reclaim_memory(value: Any = None):
    """
    Reclaim memory by deleting the given value
    and running garbage collection

    :param value: value to delete
    """
    if value is not None:
        del value
    gc.collect()
    torch.cuda.empty_cache()


def pseudo_quantize_tensor(
    weights: Tensor,
    symmetric: bool = False,
    bit_width: int = 8,
    group_size: int = -1,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Quantize the given weights to the given bit width

    :param weights: weights to quantize
    :param symmetric: whether to use symmetric quantization
    :param bit_width: bit width to quantize the weights to
    :param group_size: group size to quantize the weights to
    :return: A tuple containing the quantized weights, scales, and zeros
    """
    org_w_shape = weights.shape
    if group_size > 0:
        assert org_w_shape[-1] % group_size == 0
        weights = weights.reshape(-1, group_size)
    assert weights.dim() == 2
    assert torch.isnan(weights).sum() == 0

    if not symmetric:
        max_val = weights.amax(dim=1, keepdim=True)
        min_val = weights.amin(dim=1, keepdim=True)
        max_int = 2**bit_width - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        weights = (
            torch.clamp(torch.round(weights / scales) + zeros, min_int, max_int) - zeros
        ) * scales
        zeros = zeros.view(org_w_shape[0], -1)
    else:
        max_val = weights.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (bit_width - 1) - 1
        min_int = -(2 ** (bit_width - 1))
        scales = max_val / max_int
        zeros = None
        weights = torch.clamp(torch.round(weights / scales), min_int, max_int) * scales

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(weights).sum() == 0

    scales = scales.view(org_w_shape[0], -1)
    weights = weights.reshape(org_w_shape)

    return weights, scales, zeros


def pseudo_dequantize_tensor(
    weights: Tensor,
    scales: Tensor,
    zeros: Optional[Tensor] = None,
    symmetric: bool = False,
) -> Tensor:
    """
    Dequantize the given weights using the given scales and zeros

    :param weights: weights to dequantize
    :param scales: scales to use for dequantization
    :param zeros: zeros to use for dequantization
    :param symmetric: whether to use symmetric quantization
    :return: Dequantized weights
    """
    repeat_count = weights.data.shape[-1] // scales.shape[-1]
    scales = scales.repeat(1, repeat_count).reshape(weights.data.shape)

    if not symmetric:
        zeros = zeros.repeat(1, repeat_count).reshape(weights.data.shape)
        weights = (weights.data - zeros) * scales
    else:
        weights = weights.data * scales

    return weights
