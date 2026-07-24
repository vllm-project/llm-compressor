import math
from copy import copy

import torch
import transformers
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    fake_quantize,
)
from loguru import logger

try:
    import triton
    import triton.language as tl

    _triton_available = True
except ImportError:
    _triton_available = False

GPTQ_PRECISION = torch.float32

__all__ = ["make_empty_hessian", "accumulate_hessian", "quantize_weight"]


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


def _get_block_shape(strategy, quant_args, num_rows, num_columns):
    if strategy == QuantizationStrategy.TENSOR:
        return num_rows, num_columns
    elif strategy == QuantizationStrategy.CHANNEL:
        return 1, num_columns
    elif strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        return 1, quant_args.group_size
    elif strategy == QuantizationStrategy.BLOCK:
        return quant_args.block_structure
    else:
        raise ValueError(f"Unsupported strategy for Triton GPTQ: {strategy}")


def _build_codebook(scale, zero_point, quant_args, global_scale=None):
    num_bits = quant_args.num_bits
    if quant_args.symmetric:
        q_min = -(2 ** (num_bits - 1))
        q_max = 2 ** (num_bits - 1) - 1
    else:
        q_min = 0
        q_max = 2**num_bits - 1

    int_vals = torch.arange(
        q_min, q_max + 1, device=scale.device, dtype=torch.float32
    )

    s = scale.to(dtype=torch.float32)
    zp = zero_point.to(dtype=torch.float32)
    while s.dim() < 2:
        s = s.unsqueeze(0)
        zp = zp.unsqueeze(0)

    codebook = (int_vals - zp.unsqueeze(-1)) * s.unsqueeze(-1)

    if global_scale is not None:
        codebook = codebook / global_scale

    return codebook.contiguous()


def _build_cutoffs_and_codes(scale, zero_point, quant_args, global_scale=None):
    codes = _build_codebook(scale, zero_point, quant_args, global_scale)
    cutoffs = (codes[..., :-1] + codes[..., 1:]) * 0.5
    return codes, cutoffs


if _triton_available:

    @triton.jit
    def _quantize_block_triton_cutoff_kernel(
        W1_ptr,
        Hinv1_ptr,
        codes_ptr,
        cutoffs_ptr,
        Q1_ptr,
        Err1_ptr,
        losses1_ptr,
        num_rows,
        count,
        col_offset,
        stride_w_row,
        stride_h_row,
        stride_codes_row,
        stride_codes_col,
        stride_cut_row,
        stride_cut_col,
        r_b,
        c_b,
        num_codes,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= num_rows:
            return

        qrow = row // r_b

        cols = tl.arange(0, BLOCK_N)
        cmask = cols < count
        cut_idx = tl.arange(0, BLOCK_C)
        num_cutoffs = num_codes - 1
        cut_mask = cut_idx < num_cutoffs

        w = tl.load(W1_ptr + row * stride_w_row + cols, mask=cmask, other=0.0)

        q_out = tl.zeros([BLOCK_N], dtype=tl.float32)
        err_out = tl.zeros([BLOCK_N], dtype=tl.float32)
        loss_out = tl.zeros([BLOCK_N], dtype=tl.float32)

        for i in range(count):
            imask = cols == i
            wi = tl.sum(tl.where(imask, w, 0.0))
            d = tl.load(Hinv1_ptr + i * stride_h_row + i)

            qcol = (col_offset + i) // c_b

            cuts = tl.load(
                cutoffs_ptr
                + qrow * stride_cut_row
                + qcol * stride_cut_col
                + cut_idx,
                mask=cut_mask,
                other=float("inf"),
            )

            bin_idx = tl.sum((wi >= cuts).to(tl.int32), axis=0)

            qi = tl.load(
                codes_ptr
                + qrow * stride_codes_row
                + qcol * stride_codes_col
                + bin_idx
            )

            diff = wi - qi
            err = diff / d

            q_out = tl.where(imask, qi, q_out)
            err_out = tl.where(imask, err, err_out)
            loss_out = tl.where(imask, diff * diff / (d * d), loss_out)

            h_row = tl.load(
                Hinv1_ptr + i * stride_h_row + cols, mask=cmask, other=0.0
            )
            w = tl.where(cols >= i, w - err * h_row, w)

        tl.store(Q1_ptr + row * stride_w_row + cols, q_out, mask=cmask)
        tl.store(Err1_ptr + row * stride_w_row + cols, err_out, mask=cmask)
        tl.store(losses1_ptr + row * stride_w_row + cols, loss_out, mask=cmask)


def _quantize_block_triton_cutoff(
    W1, Hinv1, codes, cutoffs, count, col_offset, r_b, c_b
):
    num_rows = W1.shape[0]
    num_codes = codes.shape[2]
    Q1 = torch.zeros_like(W1)
    Err1 = torch.zeros_like(W1)
    losses1 = torch.zeros_like(W1)

    BLOCK_N = triton.next_power_of_2(count)
    BLOCK_C = triton.next_power_of_2(num_codes - 1) if num_codes > 1 else 1
    grid = (num_rows,)

    _quantize_block_triton_cutoff_kernel[grid](
        W1,
        Hinv1,
        codes,
        cutoffs,
        Q1,
        Err1,
        losses1,
        num_rows,
        count,
        col_offset,
        W1.stride(0),
        Hinv1.stride(0),
        codes.stride(0),
        codes.stride(1),
        cutoffs.stride(0),
        cutoffs.stride(1),
        r_b,
        c_b,
        num_codes,
        BLOCK_N=BLOCK_N,
        BLOCK_C=BLOCK_C,
    )

    return Q1, Err1, losses1


def quantize_weight(
    module: torch.nn.Module,
    quant_args: QuantizationArgs,
    hessian: torch.Tensor,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    Quantize a module weight according to the GPTQ algorithm

    :param module: module with weight being quantized
    :param quant_args: quantization arguments used to find quantization parameters
    :param hessian: preaccumulated hessian for quantization
    :param blocksize: chunk size of quantization updates
    :param percdamp: dampening factor on hessian diagonal
    :return: loss, quantized_weight, scale, zero_point, g_idx
    """
    strategy = quant_args.strategy
    actorder = quant_args.actorder
    final_shape = module.weight.shape
    final_dtype = module.weight.dtype
    W = module.weight.clone()
    H = hessian

    observer = module.weight_observer

    W = W.to(dtype=GPTQ_PRECISION)
    num_rows = W.shape[0]
    num_columns = W.shape[1]

    if actorder == ActivationOrdering.GROUP and strategy not in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        logger.warning(
            "ActivationOrdering.GROUP requires a grouped quantization strategy; "
            "falling back to actorder=None for this module."
        )
        actorder = None

    # handle activation ordering
    if actorder:
        W, H, perm = _apply_activation_ordering(W, H)

    # handle g_idx and activation ordering
    if actorder == ActivationOrdering.GROUP:
        # re-observe with permuted weight for correct per-group scales
        observer.delete_statistics(check_fused=False)
        observer(W)
        # use identity g_idx (invert permutation later)

    # handle g_idx
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
        g_idx = torch.arange(num_columns, device=W.device, dtype=torch.int) // divisor

        if actorder == ActivationOrdering.WEIGHT:
            g_idx = g_idx[perm]

    qparams = observer.get_qparams()
    scale, zero_point, global_scale = (
        qparams["scale"],
        qparams["zero_point"],
        qparams["global_scale"],
    )

    losses = torch.zeros(num_rows, device=module.weight.device)

    # mask dead hessian values
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # compute inverse hessian in place to save memory
    try:
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
    except torch._C._LinAlgError:
        logger.warning(
            "Failed to invert hessian due to numerical instability. Consider "
            "increasing GPTQModifier.dampening_frac, increasing the number "
            "of calibration samples, or shuffling the calibration dataset. "
            "Falling back to round-to-nearest for this module."
        )
        Hinv = H = torch.eye(num_columns, dtype=H.dtype, device=H.device)

    use_triton_block = (
        _triton_available
        and W.is_cuda
        and strategy
        in (
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.CHANNEL,
            QuantizationStrategy.GROUP,
            QuantizationStrategy.TENSOR_GROUP,
            QuantizationStrategy.BLOCK,
        )
    )

    if use_triton_block:
        r_b, c_b = _get_block_shape(strategy, quant_args, num_rows, num_columns)
        codes, cutoffs = _build_cutoffs_and_codes(
            scale, zero_point, quant_args, global_scale
        )

    # See section 3.4 of https://arxiv.org/abs/2203.07259
    for i1 in range(0, num_columns, blocksize):
        i2 = min(i1 + blocksize, num_columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Hinv1 = Hinv[i1:i2, i1:i2]

        if use_triton_block:
            Q1, Err1, losses1 = _quantize_block_triton_cutoff(
                W1, Hinv1.contiguous(), codes, cutoffs, count, i1, r_b, c_b
            )
            W[:, i1:i2] = Q1
            losses += torch.sum(losses1, 1) / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            continue

        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        losses1 = torch.zeros_like(W1)

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = w.clone()

            # quantize column
            if strategy == QuantizationStrategy.TENSOR:
                q = fake_quantize(
                    q, scale, zero_point, quant_args, global_scale=global_scale
                )
            elif strategy == QuantizationStrategy.CHANNEL:
                q = fake_quantize(
                    q,
                    scale[:, 0],
                    zero_point[:, 0],
                    quant_args,
                    global_scale=global_scale,
                )
            # apply global scale to scale quant scale
            elif strategy in (
                QuantizationStrategy.GROUP,
                QuantizationStrategy.TENSOR_GROUP,
            ):
                # get the group index for the current column
                column_idx = i1 + i
                group_index = g_idx[column_idx]

                # Since we're only applying quantization to a slice, this
                # ends up being a channelwise application
                altered_qargs = copy(quant_args)
                altered_qargs.strategy = QuantizationStrategy.CHANNEL

                q = fake_quantize(
                    q,
                    scale[:, group_index],
                    zero_point[:, group_index],
                    altered_qargs,
                    global_scale=global_scale,
                )
            elif strategy == QuantizationStrategy.BLOCK:
                column_idx = i1 + i
                block_column_idx = g_idx[column_idx]
                q = fake_quantize(
                    q.unsqueeze(1),
                    scale[:, block_column_idx : block_column_idx + 1],
                    zero_point[:, block_column_idx : block_column_idx + 1],
                    quant_args,
                    global_scale=global_scale,
                ).squeeze(1)
            else:
                raise ValueError(
                    f"Quantization strategy is not supported for GPTQ: {strategy}"
                )

            # propagate column error
            Q1[:, i] = q
            losses1[:, i] = (w - q) ** 2 / d**2

            err1 = (w - q) / d
            w1_err = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            W1[:, i:] -= w1_err
            Err1[:, i] = err1

        # propagate block error
        W[:, i1:i2] = Q1
        losses += torch.sum(losses1, 1) / 2

        w_err = Err1.matmul(Hinv[i1:i2, i2:])
        W[:, i2:] -= w_err

    if actorder:
        # restore original permutation
        invperm = torch.argsort(perm)
        W = W[:, invperm]

    W = W.reshape(final_shape).to(final_dtype)

    loss = torch.sum(losses).item()
    q_param_dict = {
        "weight": W,
        "weight_scale": scale.to(dtype=final_dtype),
        "weight_zero_point": zero_point.to(dtype=quant_args.zp_dtype),
    }
    if global_scale:
        q_param_dict["weight_global_scale"] = global_scale.to(dtype=final_dtype)
    if actorder == ActivationOrdering.GROUP:
        q_param_dict["weight_g_idx"] = g_idx[invperm]
    return (loss, q_param_dict)


def _apply_activation_ordering(
    W: torch.Tensor, H: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Permute weight and hessian in order of greatest output activations

    :param W: weight to permute
    :param H: hessian used to determine activation ordering
    :return: permuted weight, permuted hessian, permutation map
    """
    perm = torch.argsort(torch.diag(H), descending=True)
    return W[:, perm], H[perm][:, perm], perm
