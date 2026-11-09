"""Low-level Triton helpers shared by observer kernels."""

from enum import IntEnum

import torch
from compressed_tensors.quantization import QuantizationType
from compressed_tensors.quantization.utils.fp4_utils import _round_to_fp4
from compressed_tensors.utils.triton import tl, tldevice, triton, triton_req

try:
    from triton.runtime.jit import constexpr_function
except ImportError:
    constexpr_function = None


class TritonDType(IntEnum):
    """Shared Triton dtype codes; FP8 and E8M0 are used for scales only."""

    FP32 = 0
    FP16 = 1
    BF16 = 2
    FP8_E4M3 = 3
    E8M0 = 4


# Triton JIT functions require module globals to be `tl.constexpr` values.
_TRITON_FP32 = tl.constexpr(TritonDType.FP32.value)
_TRITON_FP16 = tl.constexpr(TritonDType.FP16.value)
_TRITON_BF16 = tl.constexpr(TritonDType.BF16.value)
_TRITON_FP8_E4M3 = tl.constexpr(TritonDType.FP8_E4M3.value)
_TRITON_E8M0 = tl.constexpr(TritonDType.E8M0.value)


if constexpr_function is not None:

    @constexpr_function
    def is_xpu_target():
        """Return whether Triton is compiling for the Intel XPU backend."""
        target = tl.target_info.current_target()
        return target is not None and target.backend == "xpu"

else:

    @triton.jit
    def is_xpu_target():
        return False


def observer_triton_req(observed, args, *unused_args, **unused_kwargs) -> bool:
    """Check whether observer inputs use a format supported by the shared kernels."""
    if not triton_req(observed):
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
        return TritonDType.FP32.value, torch.finfo(torch.float32).eps
    if dtype == torch.float16:
        return TritonDType.FP16.value, torch.finfo(torch.float16).eps
    if dtype == torch.bfloat16:
        return TritonDType.BF16.value, torch.finfo(torch.bfloat16).eps
    if dtype == torch.float8_e4m3fn:
        return TritonDType.FP8_E4M3.value, 0.125
    if dtype == torch.uint8:
        return TritonDType.E8M0.value, 2.0**-127
    raise ValueError(f"Unsupported Triton scale dtype: {dtype}")


@triton.jit
def round_to_bf16(value):
    """Round an FP32 value to BF16 and return it as FP32."""
    if is_xpu_target():
        # Intel Triton promotes BF16 arithmetic to FP32, even across an
        # explicit BF16 cast, so round through the FP32 bit representation.
        bits = value.to(tl.int32, bitcast=True)
        least_significant_bit = (bits >> 16) & 1
        bits = bits + 0x7FFF + least_significant_bit
        bits = bits & -65536
        return bits.to(tl.float32, bitcast=True)
    return value.to(tl.bfloat16).to(tl.float32)


@triton.jit
def round_to_fp16(value):
    """Round an FP32 value to FP16 and return it as FP32."""
    half = value.to(tl.float16)
    if is_xpu_target():
        # The identity FMA forces the Intel backend to materialize the FP16
        # value; a plain cast is optimized away in arithmetic chains.
        return tldevice.fma(
            half,
            tl.full((), 1.0, tl.float16),
            tl.full((), 0.0, tl.float16),
        ).to(tl.float32)
    return half.to(tl.float32)


@triton.jit
def round_fp32_as_dtype(value, COMPUTE_DTYPE: tl.constexpr):
    """Round an FP32 value to the compute dtype and return it as FP32."""
    tl.static_assert(value.dtype == tl.float32, "value must be FP32")
    if COMPUTE_DTYPE == _TRITON_FP16:
        return round_to_fp16(value)
    if COMPUTE_DTYPE == _TRITON_BF16:
        return round_to_bf16(value)
    return value


@triton.jit
def round_candidate_scale(
    scale, SCALE_ROUND_TYPE: tl.constexpr, QUANT_TYPE: tl.constexpr
):
    """Round a floating candidate scale as compressed-tensors qparams do."""
    if SCALE_ROUND_TYPE == _TRITON_FP16:
        return tl.minimum(scale, 65504.0).to(tl.float16).to(tl.float32)
    if SCALE_ROUND_TYPE == _TRITON_BF16:
        return scale.to(tl.bfloat16).to(tl.float32)
    if SCALE_ROUND_TYPE == _TRITON_FP8_E4M3:
        return tl.minimum(scale, 448.0).to(tl.float8e4nv).to(tl.float32)
    if SCALE_ROUND_TYPE == _TRITON_E8M0:
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
    q_min,
    q_max,
    SCALE_ROUND_TYPE: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Derive and round a candidate scale from candidate min/max bounds."""
    if SCALE_ROUND_TYPE == _TRITON_E8M0:
        # MX/E8M0 operates on the group absmax before element-format scaling.
        return round_candidate_scale(
            tl.maximum(tl.abs(min_value), tl.abs(max_value)),
            SCALE_ROUND_TYPE=SCALE_ROUND_TYPE,
            QUANT_TYPE=QUANT_TYPE,
        )
    if SYMMETRIC:
        scale = tl.maximum(tl.abs(min_value), tl.abs(max_value)) / (
            (q_max - q_min) * 0.5
        )
    else:
        scale = (max_value - min_value) / (q_max - q_min)
    scale = round_candidate_scale(
        scale, SCALE_ROUND_TYPE=SCALE_ROUND_TYPE, QUANT_TYPE=QUANT_TYPE
    )
    if SCALE_ROUND_TYPE == _TRITON_FP32:
        return round_fp32_as_dtype(scale, COMPUTE_DTYPE)
    return scale


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
    USE_FP8_E4B15: tl.constexpr = False,
):
    """GPTQ-style QDQ shared by MSE and importance-weighted observer kernels."""
    normalized = tldevice.div_rn(values, scale)
    normalized = round_fp32_as_dtype(normalized, COMPUTE_DTYPE)
    if HAS_ZP:
        # fake_quantize adds the zero point in the input dtype
        normalized += zero_point
        normalized = round_fp32_as_dtype(normalized, COMPUTE_DTYPE)
    normalized = tl.clamp(normalized, q_min, q_max)
    if QUANT_TYPE == 0:
        rounded = tldevice.rint(normalized)
    elif NUM_BITS == 4:
        rounded = _round_to_fp4(normalized)
    else:
        if USE_FP8_E4B15:
            rounded = (normalized * 0.00390625).to(tl.float8e4b15).to(
                tl.float32
            ) * 256.0
        else:
            rounded = normalized.to(tl.float8e4nv).to(tl.float32)
    if HAS_ZP:
        rounded -= zero_point
    return tldevice.mul_rn(rounded, scale)


@triton.jit
def calculate_qdq_error(
    values,
    quantized,
    norm: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Calculate the per-element powered QDQ error for observer search."""
    quantized = round_fp32_as_dtype(quantized, COMPUTE_DTYPE)
    values = round_fp32_as_dtype(values, COMPUTE_DTYPE)
    diff = round_fp32_as_dtype(tl.abs(quantized - values), COMPUTE_DTYPE)
    error_norm = round_fp32_as_dtype(tl.full((), norm, tl.float32), COMPUTE_DTYPE)
    if error_norm == 3.0:
        # Match CUDA eager's low-precision pow(x, 3) result. Triton's generic
        # pow differs, so round each multiply to the compute dtype.
        squared_diff = round_fp32_as_dtype(diff * diff, COMPUTE_DTYPE)
        return round_fp32_as_dtype(squared_diff * diff, COMPUTE_DTYPE)
    return round_fp32_as_dtype(tldevice.pow(diff, error_norm), COMPUTE_DTYPE)


@triton.jit
def calculate_observer_error(
    values,
    quantized,
    importance,
    group_absmax,
    norm: tl.constexpr,
    HAS_IMPORTANCE: tl.constexpr,
    HAS_MAGNITUDE_NORMALIZATION: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Calculate MSE or importance-weighted QDQ error for observer search."""
    error = calculate_qdq_error(
        values,
        quantized,
        norm,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
    )
    if HAS_IMPORTANCE:
        error = error * importance
    if HAS_MAGNITUDE_NORMALIZATION:
        error = error * tl.sqrt(values * values + group_absmax * group_absmax)
    return error
