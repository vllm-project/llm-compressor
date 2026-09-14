"""Hierarchical range search used by the expanded NVFP4 MSE observer.

The search evaluates 32 equally spaced factors over 0.8--1.8, retains the
four lowest-error factors independently for each qparam, then evaluates eight
children around each retained factor.  It therefore evaluates 64 candidates
per qparam while retaining a substantially finer effective resolution than a
linear 64-point search.
"""

import torch
from compressed_tensors.quantization import QuantizationStrategy, QuantizationType
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams, calculate_range
from compressed_tensors.quantization.utils.fp4_utils import _round_to_fp4
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import tl, tldevice, triton, triton_req

from llmcompressor.observers.base import MinMaxTuple, Observer

__all__ = ["HierarchicalMSEObserver"]


MIN_FACTOR = 0.8
MAX_FACTOR = 1.8
COARSE_POINTS = 32
RETAINED_POINTS = 4
CHILDREN_PER_POINT = 8
TILE_VALUES = 512


@triton.jit
def _calculate_candidate_scale(minimum, maximum, factor, q_min, q_max):
    return tl.maximum(tl.abs(minimum * factor), tl.abs(maximum * factor)) / (
        (q_max - q_min) * 0.5
    )


@triton.jit
def _quantize_dequantize(values, scale, q_min, q_max, OBSERVED_DTYPE: tl.constexpr):
    normalized = tldevice.div_rn(values, scale)
    if OBSERVED_DTYPE == 1:
        normalized = normalized.to(tl.float16).to(tl.float32)
    elif OBSERVED_DTYPE == 2:
        normalized = normalized.to(tl.bfloat16).to(tl.float32)
    normalized = tl.clamp(normalized, q_min, q_max)
    return tldevice.mul_rn(_round_to_fp4(normalized), scale)


def _calculate_error(
    observed: torch.Tensor,
    args,
    token_args,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    norm: float,
) -> torch.Tensor:
    scale, zero_point = calculate_qparams(minimum, maximum, args, None)
    quantized = fake_quantize(
        observed, scale.unsqueeze(-1), zero_point.unsqueeze(-1), token_args
    ).to(observed.dtype)
    return (quantized - observed).abs().pow(norm).sum(dim=(0, -1))


def _insert_best(
    errors: torch.Tensor,
    factors: torch.Tensor,
    error: torch.Tensor,
    factor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    errors = torch.cat((errors, error.unsqueeze(0)), dim=0)
    factors = torch.cat((factors, factor.unsqueeze(0)), dim=0)
    errors, indices = torch.topk(errors, RETAINED_POINTS, dim=0, largest=False)
    return errors, torch.gather(factors, 0, indices)


@ImplBackend.entrypoint("_hierarchical_search_mse")
@torch.no_grad()
def _hierarchical_search_mse(
    observed: torch.Tensor,
    args,
    token_args,
    norm: float,
) -> MinMaxTuple:
    """Eager implementation of the fixed hierarchical NVFP4 range search."""
    # NVFP4 range selection deliberately searches unrounded group scales. The
    # final compressed-tensors lifecycle still applies its configured FP8 scale.
    if args.strategy == QuantizationStrategy.TENSOR_GROUP and args.scale_dtype:
        args = args.model_copy(update={"scale_dtype": None})
        token_args = token_args.model_copy(update={"scale_dtype": None})

    minimum = torch.amin(observed, dim=(0, -1))
    maximum = torch.amax(observed, dim=(0, -1))
    shape = (RETAINED_POINTS, *minimum.shape)
    best_errors = torch.full(shape, torch.inf, device=observed.device)
    best_factors = torch.full(shape, MIN_FACTOR, device=observed.device)
    spacing = (MAX_FACTOR - MIN_FACTOR) / (COARSE_POINTS - 1)

    for index in range(COARSE_POINTS):
        factor = torch.full_like(minimum, MIN_FACTOR + index * spacing)
        error = _calculate_error(
            observed, args, token_args, minimum * factor, maximum * factor, norm
        )
        best_errors, best_factors = _insert_best(
            best_errors, best_factors, error, factor
        )

    selected_error = best_errors[0]
    selected_factor = best_factors[0]
    for parent in range(RETAINED_POINTS):
        for child in range(CHILDREN_PER_POINT):
            offset = (
                (2 * child + 1 - CHILDREN_PER_POINT)
                * spacing
                / (2 * CHILDREN_PER_POINT)
            )
            factor = (best_factors[parent] + offset).clamp(MIN_FACTOR, MAX_FACTOR)
            error = _calculate_error(
                observed, args, token_args, minimum * factor, maximum * factor, norm
            )
            improved = error < selected_error
            selected_error = torch.where(improved, error, selected_error)
            selected_factor = torch.where(improved, factor, selected_factor)

    return minimum * selected_factor, maximum * selected_factor


@triton.jit
def _insert_best_triton(error, factor, e0, p0, e1, p1, e2, p2, e3, p3):
    swap = error < e0
    error, e0 = tl.where(swap, e0, error), tl.where(swap, error, e0)
    factor, p0 = tl.where(swap, p0, factor), tl.where(swap, factor, p0)
    swap = error < e1
    error, e1 = tl.where(swap, e1, error), tl.where(swap, error, e1)
    factor, p1 = tl.where(swap, p1, factor), tl.where(swap, factor, p1)
    swap = error < e2
    error, e2 = tl.where(swap, e2, error), tl.where(swap, error, e2)
    factor, p2 = tl.where(swap, p2, factor), tl.where(swap, factor, p2)
    swap = error < e3
    return (
        e0,
        p0,
        e1,
        p1,
        e2,
        p2,
        tl.where(swap, error, e3),
        tl.where(swap, factor, p3),
    )


@triton.jit
def _candidate_error(
    values,
    value_mask,
    minimum,
    maximum,
    factor,
    q_min,
    q_max,
    norm,
    OBSERVED_DTYPE: tl.constexpr,
):
    scale = _calculate_candidate_scale(minimum, maximum, factor, q_min, q_max)
    scale = tl.maximum(scale, 1.1754943508222875e-38)
    quantized = _quantize_dequantize(
        values,
        scale[:, None],
        q_min,
        q_max,
        OBSERVED_DTYPE=OBSERVED_DTYPE,
    )
    if OBSERVED_DTYPE == 1:
        difference = quantized.to(tl.float16) - values.to(tl.float16)
    elif OBSERVED_DTYPE == 2:
        difference = quantized.to(tl.bfloat16) - values.to(tl.bfloat16)
    else:
        difference = quantized - values
    error = tl.extra.cuda.libdevice.pow(tl.abs(difference).to(tl.float32), norm)
    return tl.sum(tl.where(value_mask, error, 0.0), axis=1)


@triton.jit
def _hierarchical_search_mse_kernel(
    observed_ptr,
    minimum_ptr,
    maximum_ptr,
    factors_ptr,
    num_observations,
    num_qparams,
    group_size,
    q_min,
    q_max,
    norm,
    P_MIN: tl.constexpr,
    P_MAX: tl.constexpr,
    NUM_COARSE: tl.constexpr,
    NUM_RETAINED: tl.constexpr,
    NUM_CHILDREN: tl.constexpr,
    BLOCK_VALUES: tl.constexpr,
    TILE_QPARAMS: tl.constexpr,
    OBSERVED_DTYPE: tl.constexpr,
):
    tile = tl.arange(0, TILE_QPARAMS)
    qparams = tl.program_id(0) * TILE_QPARAMS + tile
    qparam_mask = qparams < num_qparams
    offsets = tl.arange(0, BLOCK_VALUES)
    observations = offsets // group_size
    group_offsets = offsets % group_size
    value_mask = qparam_mask[:, None] & (
        offsets[None, :] < num_observations * group_size
    )
    values = tl.load(
        observed_ptr
        + observations[None, :] * num_qparams * group_size
        + qparams[:, None] * group_size
        + group_offsets[None, :],
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    minimum = tl.load(minimum_ptr + qparams, mask=qparam_mask, other=0.0).to(tl.float32)
    maximum = tl.load(maximum_ptr + qparams, mask=qparam_mask, other=0.0).to(tl.float32)

    e0 = tl.full((TILE_QPARAMS,), float("inf"), tl.float32)
    e1, e2, e3 = e0, e0, e0
    p0 = tl.full((TILE_QPARAMS,), P_MIN, tl.float32)
    p1, p2, p3 = p0, p0, p0
    spacing: tl.constexpr = (P_MAX - P_MIN) / (NUM_COARSE - 1)
    for coarse in range(NUM_COARSE):
        factor = P_MIN + coarse * spacing
        error = _candidate_error(
            values,
            value_mask,
            minimum,
            maximum,
            factor,
            q_min,
            q_max,
            norm,
            OBSERVED_DTYPE=OBSERVED_DTYPE,
        )
        e0, p0, e1, p1, e2, p2, e3, p3 = _insert_best_triton(
            error, factor, e0, p0, e1, p1, e2, p2, e3, p3
        )

    best_error, best_factor = e0, p0
    for parent_index in range(NUM_RETAINED):
        parent = tl.where(
            parent_index == 0,
            p0,
            tl.where(parent_index == 1, p1, tl.where(parent_index == 2, p2, p3)),
        )
        for child in range(NUM_CHILDREN):
            offset: tl.constexpr = (
                (2 * child + 1 - NUM_CHILDREN) * spacing / (2 * NUM_CHILDREN)
            )
            factor = tl.maximum(P_MIN, tl.minimum(P_MAX, parent + offset))
            error = _candidate_error(
                values,
                value_mask,
                minimum,
                maximum,
                factor,
                q_min,
                q_max,
                norm,
                OBSERVED_DTYPE=OBSERVED_DTYPE,
            )
            improved = error < best_error
            best_error = tl.where(improved, error, best_error)
            best_factor = tl.where(improved, factor, best_factor)

    tl.store(factors_ptr + qparams, best_factor, mask=qparam_mask)


def _hierarchical_triton_req(observed, args, *_args, **_kwargs) -> bool:
    return (
        triton_req(observed)
        and observed.is_cuda
        and observed.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and args.type == QuantizationType.FLOAT
        and args.num_bits == 4
        and args.symmetric
        and observed.shape[0] * observed.shape[-1] <= TILE_VALUES
    )


@ImplBackend.register("_hierarchical_search_mse", _hierarchical_triton_req, 0)
@torch.no_grad()
def _hierarchical_search_mse_triton(
    observed: torch.Tensor, args, token_args, norm: float
) -> MinMaxTuple:
    """Triton implementation that packs groups into roughly 512 values/program."""
    del token_args
    if args.strategy == QuantizationStrategy.TENSOR_GROUP and args.scale_dtype:
        args = args.model_copy(update={"scale_dtype": None})

    minimum = torch.amin(observed, dim=(0, -1))
    maximum = torch.amax(observed, dim=(0, -1))
    observed = observed.contiguous().reshape(observed.shape[0], -1, observed.shape[-1])
    num_observations, num_qparams, group_size = observed.shape
    block_values = triton.next_power_of_2(num_observations * group_size)
    tile_qparams = max(1, TILE_VALUES // block_values)
    factors = torch.empty(num_qparams, dtype=torch.float32, device=observed.device)
    q_min, q_max = calculate_range(args, observed.device)
    observed_dtype = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
    }[observed.dtype]
    _hierarchical_search_mse_kernel[(triton.cdiv(num_qparams, tile_qparams),)](
        observed,
        minimum.reshape(-1).contiguous(),
        maximum.reshape(-1).contiguous(),
        factors,
        num_observations,
        num_qparams,
        group_size,
        float(q_min),
        float(q_max),
        norm,
        P_MIN=MIN_FACTOR,
        P_MAX=MAX_FACTOR,
        NUM_COARSE=COARSE_POINTS,
        NUM_RETAINED=RETAINED_POINTS,
        NUM_CHILDREN=CHILDREN_PER_POINT,
        BLOCK_VALUES=block_values,
        TILE_QPARAMS=tile_qparams,
        OBSERVED_DTYPE=observed_dtype,
    )
    factors = factors.reshape(minimum.shape).to(minimum.dtype)
    return minimum * factors, maximum * factors


@Observer.register("hierarchical_mse")
class HierarchicalMSEObserver(Observer):
    """Memoryless Lp range observer using the fixed hierarchical schedule.

    The default ``norm=2.4`` matches the MSE observer and was selected for
    expanded NVFP4 by the Llama-3.1-8B WikiText sweep.
    """

    _act_sync_dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = self.args.observer_kwargs.get("norm", 2.4)
        self._token_args = self.args.model_copy(
            update={"strategy": QuantizationStrategy.TOKEN}
        )

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        self.min_vals, self.max_vals = _hierarchical_search_mse(
            observed, self.args, self._token_args, self.norm
        )
