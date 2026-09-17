import os
from copy import copy

import torch
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
    fake_quantize,
)
from compressed_tensors.quantization.lifecycle.forward_helpers import _is_fp8_supported
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import HAS_TRITON, tl, triton, triton_req

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


def fused_gptq_block_update(
    work: torch.Tensor,
    hinv: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    quantized: torch.Tensor,
    errors: torch.Tensor,
    q_min: float,
    q_max: float,
    quant_type: int,
) -> None:
    """Run the fused Triton GPTQ block update on FP32 working tensors."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is unavailable")
    if (
        work.dim() != 3
        or work.device.type != "cuda"
        or work.dtype != torch.float32
        or hinv.dtype != torch.float32
        or quantized.dtype != torch.float32
        or errors.dtype != torch.float32
        or scale.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or any(
            tensor.device != work.device for tensor in (hinv, scale, quantized, errors)
        )
        or (zero_point is not None and zero_point.device != work.device)
    ):
        raise ValueError("invalid tensors for fused GPTQ block update")

    batch, out_rows, width = work.shape
    if (
        width <= 0
        or width > 256
        or width & (width - 1)
        or hinv.shape != (batch, width, width)
        or scale.shape != work.shape
        or quantized.shape != work.shape
        or errors.shape != work.shape
    ):
        raise ValueError("invalid shapes for fused GPTQ block update")
    if zero_point is not None and zero_point.shape != work.shape:
        raise ValueError("zero_point must have the same shape as work")

    dequant_dtype = {
        torch.float32: 0,
        torch.bfloat16: 1,
        torch.float16: 2,
    }[scale.dtype]
    has_zp = zero_point is not None
    use_fp8_e4b15 = quant_type == 2 and not _is_fp8_supported(work.device)
    if has_zp:
        zero_point = zero_point.to(torch.float32)

    block_rows = 16
    _gptq_block_update_kernel[(batch, triton.cdiv(out_rows, block_rows))](
        work,
        hinv,
        scale,
        zero_point if has_zp else scale,
        quantized,
        errors,
        out_rows,
        *work.stride(),
        *hinv.stride(),
        *scale.stride(),
        *(zero_point.stride() if has_zp else (0, 0, 0)),
        *quantized.stride(),
        *errors.stride(),
        float(q_min),
        float(q_max),
        WIDTH=width,
        QUANT_TYPE=quant_type,
        DEQUANT_DTYPE=dequant_dtype,
        HAS_ZP=has_zp,
        USE_FP8_E4B15=use_fp8_e4b15,
        BLOCK_ROWS=block_rows,
        num_warps=4,
        num_stages=2,
    )


if HAS_TRITON:

    @triton.jit
    def _gptq_block_update_kernel(
        work_ptr,
        hinv_ptr,
        scale_ptr,
        zp_ptr,
        quant_ptr,
        errors_ptr,
        out_rows,
        stride_w_b,
        stride_w_r,
        stride_w_c,
        stride_h_b,
        stride_h_r,
        stride_h_c,
        stride_s_b,
        stride_s_r,
        stride_s_c,
        stride_z_b,
        stride_z_r,
        stride_z_c,
        stride_q_b,
        stride_q_r,
        stride_q_c,
        stride_e_b,
        stride_e_r,
        stride_e_c,
        q_min,
        q_max,
        WIDTH: tl.constexpr,
        QUANT_TYPE: tl.constexpr,
        DEQUANT_DTYPE: tl.constexpr,
        HAS_ZP: tl.constexpr,
        USE_FP8_E4B15: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        batch = tl.program_id(axis=0)
        row_block = tl.program_id(axis=1)
        rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, WIDTH)
        batch_i64 = batch.to(tl.int64)
        rows_i64 = rows.to(tl.int64)
        cols_i64 = cols.to(tl.int64)
        row_mask = rows < out_rows
        work_offsets = (
            batch_i64 * stride_w_b
            + rows_i64[:, None] * stride_w_r
            + cols_i64[None, :] * stride_w_c
        )
        work = tl.load(work_ptr + work_offsets, mask=row_mask[:, None], other=0.0).to(
            tl.float32
        )

        for column in range(0, WIDTH):
            selector = cols[None, :] == column
            weight_column = tl.sum(tl.where(selector, work, 0.0), axis=1)
            scale = tl.load(
                scale_ptr
                + batch_i64 * stride_s_b
                + rows_i64 * stride_s_r
                + column * stride_s_c,
                mask=row_mask,
                other=1.0,
            ).to(tl.float32)
            scale = tl.maximum(scale, 1.1754943508222875e-38)
            normalized = tl.extra.cuda.libdevice.div_rn(weight_column, scale)
            if HAS_ZP:
                zp = tl.load(
                    zp_ptr
                    + batch_i64 * stride_z_b
                    + rows_i64 * stride_z_r
                    + column * stride_z_c,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                normalized = normalized + zp

            clamped = tl.clamp(normalized, q_min, q_max)
            if QUANT_TYPE == 0:
                rounded = tl.extra.cuda.libdevice.rint(clamped)
            elif QUANT_TYPE == 1:
                absolute = tl.abs(clamped)
                magnitude = tl.where(
                    absolute <= 0.25,
                    0.0,
                    tl.where(
                        absolute < 0.75,
                        0.5,
                        tl.where(
                            absolute <= 1.25,
                            1.0,
                            tl.where(
                                absolute < 1.75,
                                1.5,
                                tl.where(
                                    absolute <= 2.5,
                                    2.0,
                                    tl.where(
                                        absolute < 3.5,
                                        3.0,
                                        tl.where(absolute <= 5.0, 4.0, 6.0),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
                rounded = tl.where(clamped < 0.0, -magnitude, magnitude)
            else:
                # Ampere does not expose the native E4M3FN (``float8e4nv``) conversion
                # in Triton, but it does expose E4B15.  The formats have the same sign,
                # exponent, and mantissa widths and differ only by eight in exponent
                # bias, so scaling by 2**-8 before the cast and 2**8 afterwards gives
                # the E4M3FN rounding operation without requiring Hopper instructions.
                if USE_FP8_E4B15:
                    rounded = (clamped * 0.00390625).to(tl.float8e4b15).to(
                        tl.float32
                    ) * 256.0
                else:
                    rounded = clamped.to(tl.float8e4nv).to(tl.float32)

            if DEQUANT_DTYPE == 1:
                rounded = rounded.to(tl.bfloat16)
                scale_value = scale.to(tl.bfloat16)
                if HAS_ZP:
                    rounded = (rounded - zp.to(tl.bfloat16)).to(tl.bfloat16)
                quantized_column = (
                    (rounded * scale_value).to(tl.bfloat16).to(tl.float32)
                )
            elif DEQUANT_DTYPE == 2:
                rounded = rounded.to(tl.float16)
                scale_value = scale.to(tl.float16)
                if HAS_ZP:
                    rounded = (rounded - zp.to(tl.float16)).to(tl.float16)
                quantized_column = (rounded * scale_value).to(tl.float16).to(tl.float32)
            else:
                if HAS_ZP:
                    quantized_column = tl.extra.cuda.libdevice.mul_rn(
                        tl.extra.cuda.libdevice.sub_rn(rounded, zp), scale
                    )
                else:
                    quantized_column = tl.extra.cuda.libdevice.mul_rn(rounded, scale)

            diagonal = tl.load(
                hinv_ptr
                + batch_i64 * stride_h_b
                + column * stride_h_r
                + column * stride_h_c,
            ).to(tl.float32)
            error = tl.extra.cuda.libdevice.div_rn(
                tl.extra.cuda.libdevice.sub_rn(weight_column, quantized_column),
                diagonal,
            )
            q_offsets = (
                batch_i64 * stride_q_b + rows_i64 * stride_q_r + column * stride_q_c
            )
            e_offsets = (
                batch_i64 * stride_e_b + rows_i64 * stride_e_r + column * stride_e_c
            )
            tl.store(quant_ptr + q_offsets, quantized_column, mask=row_mask)
            tl.store(errors_ptr + e_offsets, error, mask=row_mask)

            hinv_row = tl.load(
                hinv_ptr
                + batch_i64 * stride_h_b
                + column * stride_h_r
                + cols_i64 * stride_h_c,
            ).to(tl.float32)
            tail = cols[None, :] > column
            update = tl.extra.cuda.libdevice.mul_rn(error[:, None], hinv_row[None, :])
            work = tl.where(tail, tl.extra.cuda.libdevice.sub_rn(work, update), work)

        tl.store(work_ptr + work_offsets, work, mask=row_mask[:, None])
