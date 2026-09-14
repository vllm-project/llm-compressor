import os
from copy import copy

import torch
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    fake_quantize,
)
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import triton_req

from llmcompressor.modifiers.gptq.gptq_triton import (
    fused_gptq_block_update,
)
from llmcompressor.modifiers.gptq.helpers import (
    GPTQ_PRECISION,
    apply_activation_ordering,
    column_scale_window,
    factorize_hessian,
    get_triton_gptq_config,
)

__all__ = [
    "quantize_weight",
]


def quantize_weight(
    weights: torch.Tensor,
    hessians: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    global_scale: torch.Tensor | None,
    quant_args: QuantizationArgs,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a batch of weights according to the GPTQ algorithm.

    :param weights: weights with shape [batch, rows, columns]
    :param hessians: normalized Hessians with shape [batch, columns, columns]
    :param scale: stacked observer scales
    :param zero_point: stacked observer zero points
    :param global_scale: optional stacked observer global scales
    :param quant_args: quantization arguments used to find quantization parameters
    :param blocksize: chunk size of quantization updates
    :param percdamp: dampening factor on hessian diagonal
    :return: quantized weights, per-batch losses, and RTN fallback flags
    """
    if weights.ndim != 3 or hessians.ndim != 3:
        raise ValueError("weights and hessians must have shape [batch, ...]")
    if weights.shape[0] != hessians.shape[0]:
        raise ValueError("weights and hessians must have matching batch sizes")

    batch_size, num_rows, num_columns = weights.shape
    strategy = quant_args.strategy
    actorder = quant_args.actorder
    final_dtype = weights.dtype
    device = weights.device
    # The caller provides a disposable stacked weight tensor, so use it as the
    # working buffer when it is already FP32 instead of allocating another copy.
    W = weights.to(device=device, dtype=GPTQ_PRECISION)
    # The stacked Hessian is the disposable working buffer.
    H = hessians.to(device=device, dtype=GPTQ_PRECISION)
    del weights, hessians
    scale = scale.to(device=device)
    zero_point = zero_point.to(device=device)
    if global_scale is not None:
        global_scale = global_scale.to(device=device)

    W, H, perm = apply_activation_ordering(W, H, quant_args.actorder)
    # handle g_idx
    g_idx = None
    if strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
        QuantizationStrategy.BLOCK,
    ):
        # mapping from column index to group index
        divisor = (
            quant_args.group_size
            if strategy != QuantizationStrategy.BLOCK
            else quant_args.block_structure[1]
        )
        g_idx = torch.arange(num_columns, device=device, dtype=torch.int) // divisor

        if actorder == ActivationOrdering.WEIGHT:
            g_idx = torch.gather(g_idx.unsqueeze(0).expand(batch_size, -1), 1, perm)

    losses = torch.zeros(batch_size, num_rows, device=device)
    used_rtn_fallback = torch.zeros(batch_size, dtype=torch.bool, device=device)
    Hinv = factorize_hessian(W, H, percdamp, used_rtn_fallback)
    # See section 3.4 of https://arxiv.org/abs/2203.07259
    for i1 in range(0, num_columns, blocksize):
        i2 = min(i1 + blocksize, num_columns)

        W1 = W[:, :, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[:, i1:i2, i1:i2]

        gptq_block_update(
            W1,
            Hinv1,
            Q1,
            Err1,
            losses1,
            scale=scale,
            zero_point=zero_point,
            global_scale=global_scale,
            g_idx=g_idx,
            quant_args=quant_args,
            i1=i1,
        )

        # propagate block error
        W[:, :, i1:i2] = Q1
        losses += torch.sum(losses1, 2) / 2

        w_err = torch.bmm(Err1, Hinv[:, i1:i2, i2:])
        W[:, :, i2:] -= w_err

    if perm is not None:
        # restore original permutation
        # Release block-local tensors before allocating the restored output.
        del W1, Q1, Err1, losses1, Hinv1, w_err
        invperm = torch.argsort(perm, dim=-1)
        W = torch.gather(W, -1, invperm.unsqueeze(-2).expand(-1, num_rows, -1))

    return W.to(final_dtype), losses.sum(dim=1), used_rtn_fallback


@ImplBackend.entrypoint("gptq_block_update")
def gptq_block_update(
    W1: torch.Tensor,
    Hinv1: torch.Tensor,
    Q1: torch.Tensor,
    Err1: torch.Tensor,
    losses1: torch.Tensor,
    *,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    quant_args: QuantizationArgs,
    i1: int,
) -> None:
    """Run one GPTQ block with the eager Torch implementation."""
    if W1.dim() != 3:
        raise ValueError("The eager GPTQ block backend requires a 3D weight block")

    block_width = W1.shape[-1]
    altered_qargs = copy(quant_args)
    altered_qargs.strategy = QuantizationStrategy.CHANNEL
    eff, zp = column_scale_window(
        scale,
        zero_point,
        global_scale,
        g_idx,
        quant_args,
        num_rows=W1.shape[-2],
        i1=i1,
        i2=i1 + block_width,
    )
    for i in range(block_width):
        w = W1[:, :, i]
        # Flatten batch into CT's column dimension, allowing one QDQ call while
        # retaining an independent scale for every batch item and output row.
        q = fake_quantize(
            w.transpose(0, 1),
            eff[:, :, i].transpose(0, 1),
            None if zp is None else zp[:, :, i].transpose(0, 1),
            altered_qargs,
        ).transpose(0, 1)

        diagonal = Hinv1[:, i, i]
        error = (w - q) / diagonal[:, None]
        Q1[:, :, i] = q
        Err1[:, :, i] = error
        losses1[:, :, i] = error.square()
        W1[:, :, i:] -= error.unsqueeze(-1) * Hinv1[:, i, i:].unsqueeze(1)


def _gptq_block_update_triton_req(
    W1: torch.Tensor,
    Hinv1: torch.Tensor,
    Q1: torch.Tensor,
    Err1: torch.Tensor,
    losses1: torch.Tensor,
    *,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    quant_args: QuantizationArgs,
    i1: int,
) -> bool:
    block_width = W1.shape[-1]
    return (
        triton_req(W1)
        and os.environ.get("LLMCOMPRESSOR_DISABLE_GPTQ_TRITON", "0") != "1"
        and get_triton_gptq_config(quant_args) is not None
        and 0 < block_width <= 256
        # Check that GPTQ block width is a power of two.
        and not block_width & (block_width - 1)
    )


@ImplBackend.register("gptq_block_update", _gptq_block_update_triton_req, 0)
def _gptq_block_update_triton(
    W1: torch.Tensor,
    Hinv1: torch.Tensor,
    Q1: torch.Tensor,
    Err1: torch.Tensor,
    losses1: torch.Tensor,
    *,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    quant_args: QuantizationArgs,
    i1: int,
) -> None:
    """Run one GPTQ block with the registered Triton backend."""
    kernel_config = get_triton_gptq_config(quant_args)
    if kernel_config is None:
        raise ValueError(f"Unsupported Triton GPTQ scheme: {quant_args}")

    block_width = W1.shape[-1]
    assert (
        0 < block_width <= 256 and block_width & (block_width - 1) == 0
    ), "Triton GPTQ block width must be a power of two <= 256"

    quant_type, q_min, q_max = kernel_config
    eff, zp = column_scale_window(
        scale,
        zero_point,
        global_scale,
        g_idx,
        quant_args,
        num_rows=W1.shape[-2],
        i1=i1,
        i2=i1 + block_width,
    )
    fused_gptq_block_update(
        W1.unsqueeze(-3) if W1.dim() == 2 else W1,
        Hinv1.unsqueeze(-3) if Hinv1.dim() == 2 else Hinv1,
        eff.unsqueeze(-3) if eff.dim() == 2 else eff,
        zp if zp is None or zp.dim() == 3 else zp.unsqueeze(-3),
        Q1.unsqueeze(-3) if Q1.dim() == 2 else Q1,
        Err1.unsqueeze(-3) if Err1.dim() == 2 else Err1,
        q_min,
        q_max,
        quant_type,
    )
    losses1.copy_(Err1.square())
