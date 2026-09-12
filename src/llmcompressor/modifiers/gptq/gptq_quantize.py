import math
import os
from copy import copy

import torch
import transformers
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
    fake_quantize,
)
from compressed_tensors.quantization.utils import calculate_range
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import triton_req

from llmcompressor.modifiers.gptq.gptq_triton import (
    FusedQuantType,
    fused_gptq_block_update,
)

GPTQ_PRECISION = torch.float32
MIN_BATCHED_CHOLESKY_SIZE = 16


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

    permuted_hessians = torch.gather(
        hessians,
        -1,
        hessian_perm.unsqueeze(-2).expand(-1, num_columns, -1),
    )
    permuted_hessians = torch.gather(
        permuted_hessians,
        -2,
        hessian_perm.unsqueeze(-1).expand(-1, -1, num_columns),
    )
    hessians.copy_(permuted_hessians)
    del permuted_hessians

    permuted_weights = torch.gather(
        weights,
        -1,
        weight_perm.unsqueeze(-2).expand(-1, num_rows, -1),
    )
    weights.copy_(permuted_weights)
    del permuted_weights
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _factorize_hessian(
    weights: torch.Tensor,
    hessians: torch.Tensor,
    percdamp: float,
    used_rtn_fallback: torch.Tensor,
) -> torch.Tensor:
    """Prepare and factorize GPTQ Hessians in place."""
    batch_size, _, num_columns = weights.shape
    diag = torch.diagonal(hessians, dim1=-2, dim2=-1)
    dead = diag == 0
    if dead.any():
        diag.masked_fill_(dead, 1.0)
        weights.masked_fill_(dead.unsqueeze(1), 0)

    # doing singletons is faster than 2 <= batches < 16
    if batch_size < MIN_BATCHED_CHOLESKY_SIZE:
        info = torch.empty((), dtype=torch.int32, device=hessians.device)
        identity = None
        for index in range(batch_size):
            damp = percdamp * torch.mean(torch.diag(hessians[index]))
            torch.diagonal(hessians[index]).add_(damp)
            torch.linalg.cholesky_ex(
                hessians[index], check_errors=False, out=(hessians[index], info)
            )
            if info.item() == 0:
                torch.cholesky_inverse(hessians[index], out=hessians[index])
                torch.linalg.cholesky(hessians[index], upper=True, out=hessians[index])
            else:
                if identity is None:
                    identity = torch.eye(
                        num_columns, dtype=hessians.dtype, device=hessians.device
                    )
                hessians[index].copy_(identity)
                used_rtn_fallback[index] = True
        return hessians

    damp = percdamp * diag.mean(dim=-1)
    diag.add_(damp.unsqueeze(-1))
    info = torch.empty(batch_size, dtype=torch.int32, device=hessians.device)
    torch.linalg.cholesky_ex(hessians, check_errors=False, out=(hessians, info))
    bad = info.nonzero(as_tuple=False).flatten()
    if bad.numel():
        hessians.index_copy_(
            0,
            bad,
            torch.eye(num_columns, dtype=hessians.dtype, device=hessians.device).expand(
                bad.numel(), -1, -1
            ),
        )
        used_rtn_fallback[bad] = True
    torch.cholesky_inverse(hessians, out=hessians)
    torch.linalg.cholesky(hessians, upper=True, out=hessians)
    return hessians


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

    W, H, perm = _apply_activation_ordering(W, H, quant_args.actorder)
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
    Hinv = _factorize_hessian(W, H, percdamp, used_rtn_fallback)
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


def _get_triton_gptq_config(
    quant_args: QuantizationArgs,
) -> tuple[int, float, float] | None:
    """
    Resolve fused GPTQ kernel configuration, or None if the scheme is not
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
        eff = scale.reshape(-1, 1, 1)
        if has_zp:
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

    if global_scale is not None:
        gs = global_scale
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
        and _get_triton_gptq_config(quant_args) is not None
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
    kernel_config = _get_triton_gptq_config(quant_args)
    if kernel_config is None:
        raise ValueError(f"Unsupported Triton GPTQ scheme: {quant_args}")

    block_width = W1.shape[-1]
    assert (
        0 < block_width <= 256 and block_width & (block_width - 1) == 0
    ), "Triton GPTQ block width must be a power of two <= 256"

    quant_type, q_min, q_max = kernel_config
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
    altered_qargs = copy(quant_args)
    altered_qargs.strategy = QuantizationStrategy.CHANNEL
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
