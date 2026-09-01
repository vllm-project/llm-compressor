"""
Fused Triton kernel for the GPTQ inner column-update loop.

The eager GPTQ loop quantizes one column at a time and issues a rank-1 update
of the shrinking unquantized suffix per column, which launches several small
kernels per column. This module fuses a whole block of columns into one
kernel: each program keeps a tile of output rows for the entire block in
registers, quantizes columns sequentially, and applies the error-propagation
updates in place.

The quantization math must match `compressed_tensors.quantization.fake_quantize`
(quantize-then-dequantize) bit-for-bit at non-tie inputs:

    eff_scale = scale / global_scale            (folded in on the host side)
    q = round(clamp(w / eff_scale + zp))        (rint for int, E2M1 for fp4)
    dequant = (q - zp) * eff_scale

The FP4 rounding boundaries replicate
`compressed_tensors.quantization.utils.fp4_utils.cast_to_fp4`
(<= 0.25 -> 0, [0.75, 1.25] -> 1.0, <= 2.5 -> 2.0, <= 5.0 -> 4.0, ...).

"""

import torch
from compressed_tensors.utils.triton import HAS_TRITON, tl, triton

__all__ = ["fused_gptq_block_update", "FusedQuantType"]


class FusedQuantType:
    INT = 0
    FP4_E2M1 = 1
    FP8_E4M3 = 2


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
        HAS_ZP: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        batch = tl.program_id(axis=0)
        row_block = tl.program_id(axis=1)
        rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, WIDTH)
        # Use int64 addressing: batched expert Hessian slabs can exceed 2**31
        # elements (e.g. 32 experts x 8192 x 8192), overflowing int32 offsets.
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

        # NOTE: device-side loop, deliberately NOT tl.static_range — fully
        # unrolling WIDTH=128 columns makes Triton/LLVM compilation take
        # minutes. A rolled loop compiles in seconds with negligible runtime
        # difference for this memory-bound body.
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
            # guard against degenerate zero scales
            scale = tl.maximum(scale, 1.1754943508222875e-38)

            # NOTE: all arithmetic below uses explicit round-to-nearest
            # libdevice ops so results match the eager PyTorch path
            # bit-for-bit: Triton's default fp32 division is approximate, and
            # mul+sub pairs would otherwise be fused into FMAs, diverging
            # from torch's separate correctly-rounded kernels.
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
                # int: round half to even, matching torch.round
                rounded = tl.extra.cuda.libdevice.rint(clamped)
            elif QUANT_TYPE == 1:
                # FP4 E2M1, boundaries match cast_to_fp4 in compressed_tensors
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
                # E4M3FN round-to-nearest, matching torch.float8_e4m3fn.
                rounded = clamped.to(tl.float8e4nv).to(tl.float32)

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
    """
    Run the fused GPTQ block update.

    :param work: fp32 working weights, [B, out_rows, width]; updated in place
    :param hinv: fp32 upper-Cholesky inverse Hessian factor, [B, width, width]
    :param scale: effective per-column scales (global scale folded in),
        [B, out_rows, width]
    :param zero_point: optional per-column zero points, [B, out_rows, width]
    :param quantized: fp32 output buffer for dequantized columns, same shape as
        work; written in place
    :param errors: fp32 output buffer for per-column errors, same shape as
        work; written in place
    :param q_min: minimum of the quantization grid (post-scale)
    :param q_max: maximum of the quantization grid (post-scale)
    :param quant_type: FusedQuantType.INT, FP4_E2M1, or FP8_E4M3
    Invalid inputs and Triton compilation or launch failures are raised to the
    caller.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is unavailable")

    if (
        work.dim() != 3
        or work.device.type != "cuda"
        or work.dtype != torch.float32
        or hinv.dtype != torch.float32
        or quantized.dtype != torch.float32
        or errors.dtype != torch.float32
        or scale.dtype not in (torch.float32, torch.float64)
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

    scale = scale.to(torch.float32)
    has_zp = zero_point is not None
    if has_zp:
        zero_point = zero_point.to(torch.float32)

    block_rows = 16
    _gptq_block_update_kernel[(batch, triton.cdiv(out_rows, block_rows))](
        work,
        hinv,
        scale,
        zero_point if has_zp else scale,  # dummy pointer when unused
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
        HAS_ZP=has_zp,
        BLOCK_ROWS=block_rows,
        num_warps=4,
        num_stages=2,
    )
