import math
import os

import torch
import transformers
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils import calculate_range, cast_to_fp4
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import triton_req

from llmcompressor.modifiers.gptq.gptq_triton import (
    FusedQuantType,
    fused_gptq_block_update,
)

GPTQ_PRECISION = torch.float32

__all__ = [
    "make_empty_hessian",
    "accumulate_hessian",
    "quantize_weight",
]


def _apply_activation_ordering(
    weights: torch.Tensor,
    hessians: torch.Tensor,
    actorder: ActivationOrdering | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Apply GPTQ activation ordering to a weight/Hessian batch."""
    if not actorder:
        return weights, hessians, None
    if actorder not in (ActivationOrdering.WEIGHT, ActivationOrdering.STATIC):
        raise ValueError(
            f"Invalid activation ordering {actorder}. Only 'weight' and 'static'"
            " are supported for GPTQ."
        )

    num_rows, num_columns = weights.shape[-2:]
    perm = torch.argsort(
        torch.diagonal(hessians, dim1=-2, dim2=-1), dim=-1, descending=True
    )
    hessian_perm = perm
    weight_perm = perm.to(device=weights.device)
    if hessian_perm.device != hessians.device:
        hessian_perm = hessian_perm.to(device=hessians.device)

    hessians = torch.gather(
        hessians,
        -1,
        hessian_perm.unsqueeze(-2).expand(-1, num_columns, -1),
    )
    hessians = torch.gather(
        hessians,
        -2,
        hessian_perm.unsqueeze(-1).expand(-1, -1, num_columns),
    )
    weights = torch.gather(
        weights,
        -1,
        weight_perm.unsqueeze(-2).expand(-1, num_rows, -1),
    )
    return weights, hessians, weight_perm


def make_empty_hessian(
    module: torch.nn.Module, device: torch.device | None = None
) -> torch.Tensor:
    weight = module.weight
    num_columns = weight.shape[1]
    device = device if device is not None else weight.device
    return torch.zeros((num_columns, num_columns), device=device, dtype=GPTQ_PRECISION)


def accumulate_hessian(
    inp: torch.Tensor,
    module: torch.nn.Module,
    H: torch.Tensor | None,
    num_samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inp = inp.to(device=H.device)
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)
    elif len(inp.shape) > 3:
        inp = inp.reshape(inp.shape[0], -1, inp.shape[-1])

    num_added = inp.shape[0]

    match module:
        case torch.nn.Linear() | transformers.Conv1D():
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        case torch.nn.Conv2d():
            unfold = torch.nn.Unfold(
                module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

    num_samples += num_added

    inp = inp.to(dtype=GPTQ_PRECISION)
    inp = math.sqrt(2) * inp
    H += inp.matmul(inp.t())

    return H, num_samples


def quantize_weight(
    weights: torch.Tensor,
    hessians: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    global_scale: torch.Tensor | None,
    quant_args: QuantizationArgs,
    perm: torch.Tensor | None = None,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a batch of weights according to the GPTQ algorithm.

    :param weights: weights with shape [batch, rows, columns]
    :param hessians: normalized Hessians with shape [batch, columns, columns]
    :param scale: stacked observer scales
    :param zero_point: stacked observer zero points
    :param global_scale: optional stacked observer global scales
    :param quant_args: quantization arguments used to find quantization parameters
    :param perm: activation-order permutation already applied to ``weights`` and
        ``hessians``. The returned weights are restored to their original order.
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
    if actorder and perm is None:
        raise ValueError("actorder requires pre-permuted weights, hessians, and perm")
    final_dtype = weights.dtype
    device = weights.device
    # The caller provides a disposable stacked weight tensor, so use it as the
    # working buffer when it is already FP32 instead of allocating another copy.
    W = weights.to(device=device, dtype=GPTQ_PRECISION)
    # The stacked Hessian is the disposable working buffer. The caller retains
    # the original per-module Hessians separately for batch fallback.
    H = hessians.to(device=device, dtype=GPTQ_PRECISION)
    scale = scale.to(device=device)
    zero_point = zero_point.to(device=device)
    if global_scale is not None:
        global_scale = global_scale.to(device=device)

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

    # mask dead hessian values
    diag = torch.diagonal(H, dim1=-2, dim2=-1)
    dead = diag == 0
    if dead.any():
        torch.diagonal(H, dim1=-2, dim2=-1).masked_fill_(dead, 1.0)
        W.masked_fill_(dead.unsqueeze(1), 0)

    # compute inverse hessian in place to save memory
    damp = percdamp * torch.diagonal(H, dim1=-2, dim2=-1).mean(dim=-1)
    torch.diagonal(H, dim1=-2, dim2=-1).add_(damp.unsqueeze(-1))
    info = torch.empty(batch_size, dtype=torch.int32, device=device)
    torch.linalg.cholesky_ex(H, check_errors=False, out=(H, info))
    bad = info.nonzero(as_tuple=False).flatten()
    if bad.numel() and batch_size > 1:
        raise torch.linalg.LinAlgError("batched GPTQ Hessian inversion failed")
    if bad.numel():
        H[bad[0]].copy_(torch.eye(num_columns, dtype=H.dtype, device=device))
        used_rtn_fallback[bad[0]] = True
    else:
        torch.cholesky_inverse(H, out=H)
        torch.linalg.cholesky(H, upper=True, out=H)
    Hinv = H

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


def _fused_kernel_params(
    quant_args: QuantizationArgs,
) -> tuple[int, float, float] | None:
    """
    Resolve fused-kernel quantization parameters, or None if the scheme is not
    supported by the fused kernel.
    """
    if quant_args.strategy in (
        QuantizationStrategy.TENSOR,
        QuantizationStrategy.CHANNEL,
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
        QuantizationStrategy.BLOCK,
    ):
        pass
    else:
        return None

    if quant_args.type == QuantizationType.INT:
        quant_type = FusedQuantType.INT
    elif quant_args.type == QuantizationType.FLOAT and quant_args.num_bits == 4:
        quant_type = FusedQuantType.FP4_E2M1
    elif quant_args.type == QuantizationType.FLOAT and quant_args.num_bits == 8:
        quant_type = FusedQuantType.FP8_E4M3
    else:
        return None

    q_min, q_max = calculate_range(quant_args, "cpu")
    return quant_type, float(q_min), float(q_max)


def _column_scale_window(
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    quant_args: QuantizationArgs,
    num_rows: int,
    i1: int,
    i2: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Expand quantization parameters into effective per-column values over the
    column window [i1, i2), with the global scale folded in.

    Supports arbitrary leading batch dimensions: `scale` is [..., num_rows, G]
    and `g_idx` (group strategies) is [... or absent, num_columns].

    :return: (eff_scale [..., num_rows, block_width], zero_point or None)
    """
    strategy = quant_args.strategy
    block_width = i2 - i1

    has_zp = zero_point is not None and not quant_args.symmetric

    if strategy == QuantizationStrategy.TENSOR:
        # A stacked batch can be [B], [B, 1], or [B, 1, 1]. Normalize all
        # forms to [B, 1, 1] before expanding over rows and columns.
        if scale.ndim == 3:
            eff = scale
        else:
            eff = scale.reshape(-1, 1, 1)
        if has_zp:
            if zero_point.ndim == 3:
                zp = zero_point
            else:
                zp = zero_point.reshape(-1, 1, 1)
        else:
            zp = None
    elif strategy == QuantizationStrategy.CHANNEL:
        eff = scale[..., :, 0:1]
        zp = zero_point[..., :, 0:1] if has_zp else None
    elif strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        idx = g_idx[..., i1:i2].long()
        eff = torch.gather(
            scale, -1, idx.unsqueeze(-2).expand(*scale.shape[:-1], block_width)
        )
        zp = (
            torch.gather(
                zero_point,
                -1,
                idx.unsqueeze(-2).expand(*zero_point.shape[:-1], block_width),
            )
            if has_zp
            else None
        )
    elif strategy == QuantizationStrategy.BLOCK:
        block_height, _ = quant_args.block_structure
        row_idx = torch.arange(num_rows, device=scale.device) // block_height
        col_idx = g_idx[..., i1:i2].long().unsqueeze(-2)
        eff = torch.gather(
            scale,
            -1,
            col_idx.expand(*scale.shape[:-1], block_width),
        )
        row_idx = row_idx.reshape((1,) * (eff.ndim - 2) + (num_rows, 1))
        eff = torch.gather(
            eff,
            -2,
            row_idx.expand(*eff.shape[:-2], num_rows, block_width),
        )
        if has_zp:
            zp = torch.gather(
                zero_point,
                -1,
                col_idx.expand(*zero_point.shape[:-1], block_width),
            )
            zp = torch.gather(
                zp,
                -2,
                row_idx.expand(*zp.shape[:-2], num_rows, block_width),
            )
        else:
            zp = None
    else:
        raise ValueError(f"Unsupported strategy for column scale window: {strategy}")

    eff = eff.to(GPTQ_PRECISION)
    if global_scale is not None:
        gs = global_scale.to(GPTQ_PRECISION)
        gs = gs.reshape(*gs.shape, *([1] * (eff.ndim - gs.ndim)))
        eff = eff / gs
    eff = eff.expand(*eff.shape[:-2], num_rows, block_width).contiguous()

    if zp is None:
        # symmetric zero points are exactly zero; adding them is a no-op
        return eff, None

    zp = zp.to(GPTQ_PRECISION)
    zp = zp.expand(*zp.shape[:-2], num_rows, block_width).contiguous()
    return eff, zp


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
        and _fused_kernel_params(quant_args) is not None
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
    params = _fused_kernel_params(quant_args)
    if params is None:
        raise ValueError(f"Unsupported Triton GPTQ scheme: {quant_args}")

    block_width = W1.shape[-1]
    if block_width > 256 or block_width & (block_width - 1):
        raise ValueError("Triton GPTQ block width must be a power of two <= 256")

    quant_type, q_min, q_max = params
    eff, zp = _column_scale_window(
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
    q_min, q_max = calculate_range(quant_args, W1.device)
    eff, zp = _column_scale_window(
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
        normalized = w / eff[:, :, i]
        if zp is not None:
            normalized = normalized + zp[:, :, i]
        clamped = torch.clamp(normalized, q_min, q_max)
        if quant_args.type == QuantizationType.INT:
            rounded = torch.round(clamped)
        elif quant_args.num_bits == 4:
            rounded = cast_to_fp4(clamped)
        elif quant_args.num_bits == 8:
            rounded = clamped.to(torch.float8_e4m3fn).to(torch.float32)
        else:
            raise ValueError(f"Unsupported quantization scheme: {quant_args}")
        q = rounded * eff[:, :, i]
        if zp is not None:
            q = (rounded - zp[:, :, i]) * eff[:, :, i]

        diagonal = Hinv1[:, i, i]
        error = (w - q) / diagonal[:, None]
        Q1[:, :, i] = q
        Err1[:, :, i] = error
        losses1[:, :, i] = error.square()
        W1[:, :, i:] -= error.unsqueeze(-1) * Hinv1[:, i, i:].unsqueeze(1)
