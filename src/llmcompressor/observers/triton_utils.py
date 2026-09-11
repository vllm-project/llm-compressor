"""JIT-inline primitives shared by observer Triton kernels."""

from compressed_tensors.quantization.utils.fp4_utils import _round_to_fp4
from compressed_tensors.utils.triton import tl, tldevice, triton


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
):
    """GPTQ-style QDQ shared by MSE and future observer kernels."""
    normalized = tldevice.div_rn(values, scale)
    if HAS_ZP:
        normalized += zero_point
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
