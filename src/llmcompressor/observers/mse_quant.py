import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams, calculate_range
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import tl, triton, triton_req

from llmcompressor.observers.base import MinMaxTuple
from llmcompressor.observers.triton_utils import (
    calculate_candidate_scale,
    quantize_dequantize,
)


@ImplBackend.entrypoint("_grid_search_mse")
@torch.no_grad()
def _grid_search_mse(
    observed: torch.Tensor,
    args: QuantizationArgs,
    token_args: QuantizationArgs,
    maxshrink: float,
    patience: int,
    grid: float,
    norm: float,
    triton_error_buffer: float,
    expand: float = 1.0,
) -> MinMaxTuple:
    """Find per-channel min/max ranges that minimize quantization error.

    Performs a 1-D grid search over shrink factors applied to the observed
    tensor's min/max values. CUDA uses the registered Triton backend where
    supported; all other inputs retain this eager implementation.

    :param observed: value of shape (num_observations, *qparams_shape, group_size)
    :param args: quantization args used for computing qparams and fake quant
    :param token_args: quantization args with strategy set to TOKEN
    :param maxshrink: maximum shrink amount (in "grid steps"). The number of
        search steps is int(maxshrink * grid)
    :param patience: number of consecutive search steps without improvement before
        early stopping
    :param grid: resolution of the shrink search. Larger values give finer granularity
        in shrink factors
    :param norm: exponent used when computing the error. norm = 2 approximates MSE
    :param triton_error_buffer: Triton per-group patience reset threshold. A value of
        0.3 means errors within 30% of a group's best reset its counter.
    :param expand: factor to scale the initial min/max range before searching.
        Values > 1.0 let the search explore ranges wider than the observed
        values (e.g. expand=2.0 starts at 2x the observed range).
    """
    if (
        args.strategy == QuantizationStrategy.TENSOR_GROUP
        and args.scale_dtype is not None
    ):
        args = args.model_copy(update={"scale_dtype": None})
        token_args = token_args.model_copy(update={"scale_dtype": None})

    min_val = torch.amin(observed, dim=(0, -1)) * expand
    max_val = torch.amax(observed, dim=(0, -1)) * expand
    best_error = torch.full_like(min_val, torch.finfo(min_val.dtype).max)
    best_min_val = min_val.clone()
    best_max_val = max_val.clone()

    no_improve_count = 0

    # @ksayers @HGCharles: investigate searching over separate min/max
    for i in range(int(maxshrink * grid)):
        p = 1 - i / grid
        shrinked_min_val = min_val * p
        shrinked_max_val = max_val * p
        err = _calculate_error(
            observed,
            args,
            token_args,
            shrinked_min_val,
            shrinked_max_val,
            norm,
        )

        improved = err < best_error
        if torch.any(improved):
            best_error[improved] = err[improved]
            best_min_val[improved] = shrinked_min_val[improved]
            best_max_val[improved] = shrinked_max_val[improved]
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

    return best_min_val, best_max_val


@triton.jit
def _grid_search_mse_triton_kernel(
    observed_ptr,
    min_base_ptr,
    max_base_ptr,
    zp_base_ptr,
    best_step_ptr,
    num_observations,
    num_qparams,
    group_size,
    total_steps,
    inv_grid,
    q_min,
    q_max,
    norm,
    patience,
    triton_error_buffer,
    scale_eps,
    BLOCK_VALUES: tl.constexpr,
    TOTAL_STEPS: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    NUM_BITS: tl.constexpr,
    HAS_ZP: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    SCALE_ROUND_TYPE: tl.constexpr,
    OBSERVED_DTYPE: tl.constexpr,
):
    """Base-scale GPTQ search with buffered, per-qparam patience."""
    qparam = tl.program_id(0)
    min_base = tl.load(min_base_ptr + qparam).to(tl.float32)
    max_base = tl.load(max_base_ptr + qparam).to(tl.float32)
    zp = tl.load(zp_base_ptr + qparam).to(tl.float32)
    best_error = tl.full([], float("inf"), tl.float32)
    best_step = 0
    stale = 0
    total_values = num_observations * group_size

    for step in range(TOTAL_STEPS):
        if step < total_steps:
            if stale < patience:
                p = 1.0 - step * inv_grid
                scale = calculate_candidate_scale(
                    min_base,
                    max_base,
                    p,
                    q_min,
                    q_max,
                    SCALE_ROUND_TYPE=SCALE_ROUND_TYPE,
                    QUANT_TYPE=QUANT_TYPE,
                    SYMMETRIC=SYMMETRIC,
                )
                scale = tl.maximum(scale, scale_eps)
                error = tl.zeros([], tl.float32)
                for value_offset in range(0, total_values, BLOCK_VALUES):
                    offsets = value_offset + tl.arange(0, BLOCK_VALUES)
                    obs_idx = offsets // group_size
                    group_idx = offsets % group_size
                    mask = offsets < total_values
                    values = tl.load(
                        observed_ptr
                        + obs_idx * num_qparams * group_size
                        + qparam * group_size
                        + group_idx,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                    quantized = quantize_dequantize(
                        values,
                        scale,
                        zp,
                        q_min,
                        q_max,
                        QUANT_TYPE=QUANT_TYPE,
                        NUM_BITS=NUM_BITS,
                        HAS_ZP=HAS_ZP,
                    )
                    if OBSERVED_DTYPE == 1:
                        diff = tl.abs(quantized.to(tl.float16) - values.to(tl.float16))
                    elif OBSERVED_DTYPE == 2:
                        diff = tl.abs(
                            quantized.to(tl.bfloat16) - values.to(tl.bfloat16)
                        )
                    else:
                        diff = tl.abs(quantized - values)
                    diff_pow = tl.exp(tl.log(diff.to(tl.float32)) * norm)
                    error += tl.sum(tl.where(mask, diff_pow, 0.0))

                is_better = error < best_error
                within_buffer = error <= best_error * (1.0 + triton_error_buffer)
                best_error = tl.where(is_better, error, best_error)
                best_step = tl.where(is_better, step, best_step)
                stale = tl.where(within_buffer, 0, stale + 1)

    tl.store(best_step_ptr + qparam, best_step)


def _scale_round_config(dtype: torch.dtype | None) -> tuple[int, float]:
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
    raise ValueError(f"Unsupported MSE Triton scale dtype: {dtype}")


def _mse_triton_req(
    observed: torch.Tensor, args: QuantizationArgs, *unused_args, **unused_kwargs
) -> bool:
    """Limit the CUDA backend to formats represented by the shared QDQ helper."""
    if not triton_req(observed) or not observed.is_cuda:
        return False
    if observed.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if args.type == QuantizationType.INT:
        return args.num_bits <= 8
    if args.type == QuantizationType.FLOAT:
        return args.num_bits in (4, 8)
    return False


@ImplBackend.register("_grid_search_mse", _mse_triton_req, 0)
@torch.no_grad()
def _grid_search_mse_triton(
    observed: torch.Tensor,
    args: QuantizationArgs,
    token_args: QuantizationArgs,
    maxshrink: float,
    patience: int,
    grid: float,
    norm: float,
    triton_error_buffer: float,
    expand: float = 1.0,
) -> MinMaxTuple:
    """CUDA Triton implementation of buffered base-scale MSE search."""
    del token_args
    if (
        args.strategy == QuantizationStrategy.TENSOR_GROUP
        and args.scale_dtype is not None
    ):
        args = args.model_copy(update={"scale_dtype": None})

    min_val = torch.amin(observed, dim=(0, -1)) * expand
    max_val = torch.amax(observed, dim=(0, -1)) * expand
    qparam_args = (
        args.model_copy(update={"scale_dtype": None})
        if args.scale_dtype is not None
        else args
    )
    _, zp_base = calculate_qparams(min_val, max_val, qparam_args, None)
    min_base = torch.minimum(min_val, torch.zeros_like(min_val))
    min_base = min_base.reshape(-1).contiguous()
    max_base = torch.maximum(max_val, torch.zeros_like(max_val))
    max_base = max_base.reshape(-1).contiguous()
    zp_base = zp_base.reshape(-1).to(torch.float32).contiguous()

    observed = observed.contiguous().reshape(
        observed.shape[0], -1, observed.shape[-1]
    )
    num_observations, num_qparams, group_size = observed.shape
    total_steps = int(maxshrink * grid)
    best_step = torch.empty(num_qparams, dtype=torch.int32, device=observed.device)
    q_min, q_max = calculate_range(args, observed.device)
    scale_round_type, scale_eps = _scale_round_config(args.scale_dtype)
    observed_dtype = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
    }[observed.dtype]
    quant_type = 0 if args.type == QuantizationType.INT else 1
    block_values = min(1024, triton.next_power_of_2(group_size))
    _grid_search_mse_triton_kernel[(num_qparams,)](
        observed,
        min_base,
        max_base,
        zp_base,
        best_step,
        num_observations,
        num_qparams,
        group_size,
        total_steps,
        1.0 / grid,
        float(q_min),
        float(q_max),
        norm,
        patience,
        triton_error_buffer,
        scale_eps,
        BLOCK_VALUES=block_values,
        TOTAL_STEPS=triton.next_power_of_2(total_steps),
        QUANT_TYPE=quant_type,
        NUM_BITS=args.num_bits,
        HAS_ZP=not args.symmetric,
        SYMMETRIC=args.symmetric,
        SCALE_ROUND_TYPE=scale_round_type,
        OBSERVED_DTYPE=observed_dtype,
    )
    ps = 1.0 - torch.arange(
        total_steps, device=min_val.device, dtype=min_val.dtype
    ) / grid
    best_p = ps[best_step.long()].reshape(min_val.shape)
    return min_val * best_p, max_val * best_p


def _calculate_error(
    observed: torch.Tensor,
    args: QuantizationArgs,
    token_args: QuantizationArgs,
    shrinked_min: torch.Tensor,
    shrinked_max: torch.Tensor,
    norm: float,
) -> torch.Tensor:
    """Fake-quantize ``observed`` using the given shrinked min/max range and
    return the per-channel error.

    :return: per-channel quantization error, shape ``(*qparams_shape,)``
    """
    candidate_scales, candidate_zero_points = calculate_qparams(
        min_vals=shrinked_min,
        max_vals=shrinked_max,
        quantization_args=args,
        global_scale=None,
    )

    q = fake_quantize(
        observed,
        candidate_scales.unsqueeze(-1),
        candidate_zero_points.unsqueeze(-1),
        token_args,
    ).to(observed.dtype)

    err = torch.sum((q - observed).abs().pow(norm), dim=(0, -1))
    del q
    return err
