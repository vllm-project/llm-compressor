"""Low-level Triton helpers shared by observer kernels."""

import torch
from compressed_tensors.quantization import QuantizationType
from compressed_tensors.quantization.utils.fp4_utils import _round_to_fp4
from compressed_tensors.utils.triton import tl, tldevice, triton, triton_req


def observer_triton_req(observed, args, *unused_args, **unused_kwargs) -> bool:
    """Check whether observer inputs use a format supported by the shared kernels."""
    if not triton_req(observed) or not observed.is_cuda:
        return False
    if observed.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if args.type == QuantizationType.INT:
        return args.num_bits <= 8
    if args.type == QuantizationType.FLOAT:
        return args.num_bits in (4, 8)
    return False


def scale_round_config(dtype: torch.dtype | None) -> tuple[int, float]:
    """Return the Triton scale rounding mode and minimum valid scale."""
    if dtype is None or dtype == torch.float32:
        return 0, torch.finfo(torch.float32).eps
    if dtype == torch.float16:
        return 1, torch.finfo(torch.float16).eps
    if dtype == torch.bfloat16:
        return 2, torch.finfo(torch.bfloat16).eps
    if dtype == torch.float8_e4m3fn:
        return 3, 0.125
    if dtype == torch.uint8:
        return 4, 2.0**-127
    raise ValueError(f"Unsupported Triton scale dtype: {dtype}")


@triton.jit
def round_candidate_scale(
    scale, SCALE_ROUND_TYPE: tl.constexpr, QUANT_TYPE: tl.constexpr
):
    """Round a floating candidate scale as compressed-tensors qparams do."""
    if SCALE_ROUND_TYPE == 1:
        return tl.minimum(scale, 65504.0).to(tl.float16).to(tl.float32)
    if SCALE_ROUND_TYPE == 2:
        return scale.to(tl.bfloat16).to(tl.float32)
    if SCALE_ROUND_TYPE == 3:
        return tl.minimum(scale, 448.0).to(tl.float8e4nv).to(tl.float32)
    if SCALE_ROUND_TYPE == 4:
        # E8M0 stores the power-of-two scale after accounting for the element
        # format's largest exponent (FP4: 2, FP8: 8).
        is_zero = scale == 0.0
        safe_scale = tl.where(is_zero, 1.0, scale)
        exponent = tl.floor(tldevice.log2(safe_scale))
        power_2 = tldevice.exp2(exponent)
        exponent += tl.where(safe_scale >= 1.75 * power_2, 1.0, 0.0)
        offset = 2.0 if QUANT_TYPE == 1 else 8.0
        encoded = tl.clamp(exponent - offset + 127.0, 0.0, 255.0)
        encoded = tl.where(is_zero, 0.0, encoded)
        return tldevice.exp2(encoded - 127.0)
    return scale


@triton.jit
def calculate_candidate_scale(
    min_value,
    max_value,
    p,
    q_min,
    q_max,
    SCALE_ROUND_TYPE: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    SYMMETRIC: tl.constexpr,
):
    """Derive and round a candidate scale from its original min/max range."""
    scaled_min = min_value * p
    scaled_max = max_value * p
    if SCALE_ROUND_TYPE == 4:
        # MX/E8M0 operates on the group absmax before element-format scaling.
        return round_candidate_scale(
            tl.maximum(tl.abs(scaled_min), tl.abs(scaled_max)),
            SCALE_ROUND_TYPE=SCALE_ROUND_TYPE,
            QUANT_TYPE=QUANT_TYPE,
        )
    if SYMMETRIC:
        scale = tl.maximum(tl.abs(scaled_min), tl.abs(scaled_max)) / (
            (q_max - q_min) * 0.5
        )
    else:
        scale = (scaled_max - scaled_min) / (q_max - q_min)
    return round_candidate_scale(
        scale, SCALE_ROUND_TYPE=SCALE_ROUND_TYPE, QUANT_TYPE=QUANT_TYPE
    )


@triton.jit
def quantize_dequantize(
    values,
    scale,
    zero_point,
    q_min,
    q_max,
    QUANT_TYPE: tl.constexpr,
    NUM_BITS: tl.constexpr,
    HAS_ZP: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """GPTQ-style QDQ shared by MSE and importance-weighted observer kernels."""
    normalized = tldevice.div_rn(values, scale)
    if COMPUTE_DTYPE == 1:
        normalized = normalized.to(tl.float16).to(tl.float32)
    elif COMPUTE_DTYPE == 2:
        normalized = normalized.to(tl.bfloat16).to(tl.float32)
    if HAS_ZP:
        # fake_quantize adds the zero point in the input dtype
        normalized += zero_point
        if COMPUTE_DTYPE == 1:
            normalized = normalized.to(tl.float16).to(tl.float32)
        elif COMPUTE_DTYPE == 2:
            normalized = normalized.to(tl.bfloat16).to(tl.float32)
    normalized = tl.clamp(normalized, q_min, q_max)
    if QUANT_TYPE == 0:
        rounded = tldevice.rint(normalized)
    elif NUM_BITS == 4:
        rounded = _round_to_fp4(normalized)
    else:
        rounded = normalized.to(tl.float8e4nv).to(tl.float32)
    if HAS_ZP:
        rounded -= zero_point
    return tldevice.mul_rn(rounded, scale)


@triton.jit
def calculate_mse_error(
    values,
    quantized,
    norm,
    COMPUTE_DTYPE: tl.constexpr,
    ROUND_ERROR: tl.constexpr,
):
    """Calculate the per-element powered QDQ error for the MSE observer."""
    if COMPUTE_DTYPE == 1:
        diff = tl.abs(quantized.to(tl.float16) - values.to(tl.float16))
    elif COMPUTE_DTYPE == 2:
        diff = tl.abs(quantized.to(tl.bfloat16) - values.to(tl.bfloat16))
    else:
        diff = tl.abs(quantized - values)

    error_norm = norm.to(tl.float32)
    if ROUND_ERROR:
        if COMPUTE_DTYPE == 1:
            error_norm = error_norm.to(tl.float16).to(tl.float32)
        elif COMPUTE_DTYPE == 2:
            error_norm = error_norm.to(tl.bfloat16).to(tl.float32)
    diff_pow = tl.extra.cuda.libdevice.pow(diff.to(tl.float32), error_norm)
    if ROUND_ERROR:
        if COMPUTE_DTYPE == 1:
            diff_pow = diff_pow.to(tl.float16)
        elif COMPUTE_DTYPE == 2:
            diff_pow = diff_pow.to(tl.bfloat16)
    return diff_pow.to(tl.float32)


@triton.jit
def calculate_weighted_error(
    values,
    quantized,
    importance,
    norm,
    COMPUTE_DTYPE: tl.constexpr,
    ROUND_ERROR: tl.constexpr,
):
    """Calculate the powered QDQ error and apply imatrix weights."""
    diff_pow = calculate_mse_error(
        values,
        quantized,
        norm,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
        ROUND_ERROR=ROUND_ERROR,
    )
    return diff_pow * importance


@triton.jit
def calculate_observer_error(
    values,
    quantized,
    importance,
    norm,
    HAS_IMPORTANCE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    ROUND_ERROR: tl.constexpr,
):
    """Calculate MSE or importance-weighted QDQ error for observer search."""
    if HAS_IMPORTANCE:
        return calculate_weighted_error(
            values,
            quantized,
            importance,
            norm,
            COMPUTE_DTYPE=COMPUTE_DTYPE,
            ROUND_ERROR=ROUND_ERROR,
        )
    return calculate_mse_error(
        values,
        quantized,
        norm,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
        ROUND_ERROR=ROUND_ERROR,
    )
