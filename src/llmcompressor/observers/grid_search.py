"""Shared eager and Triton grid-search implementation for quantization observers."""

import math

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams, calculate_range
from compressed_tensors.utils import patch_attr
from compressed_tensors.utils.impl_backend import ImplBackend
from compressed_tensors.utils.triton import tl, triton

from llmcompressor.observers.base import MinMaxTuple
from llmcompressor.utils.triton_utils import (
    calculate_candidate_scale,
    calculate_observer_error,
    observer_triton_req,
    quantize_dequantize,
    scale_round_config,
)


def _default_triton_error_buffer(args) -> float:
    """Return the format-specific default for Triton per-group patience."""
    return 1.00 if args.type == QuantizationType.FLOAT and args.num_bits == 4 else 0.30


@torch.no_grad()
def _grid_search_observer_eager(
    observed: torch.Tensor,
    args: QuantizationArgs,
    maxshrink: float,
    patience: int,
    grid: float,
    norm: float,
    expand: float = 1.0,
    token_args: QuantizationArgs | None = None,
    importance_weights: torch.Tensor | None = None,
    use_imatrix_error: bool = False,
) -> MinMaxTuple:
    """Search candidate ranges with shared updates and selectable error arithmetic."""
    min_vals = torch.amin(observed, dim=(0, -1)) * expand
    max_vals = torch.amax(observed, dim=(0, -1)) * expand

    total_values = observed.shape[0] * observed.shape[-1]
    if use_imatrix_error:
        if total_values <= 512 and observed.dtype in (torch.float16, torch.bfloat16):
            error_dtype = observed.dtype
        else:
            error_dtype = torch.float32
        error_norm = norm
        if error_dtype != torch.float32:
            error_norm = torch.tensor(norm, dtype=error_dtype).item()
    else:
        if token_args is None:
            raise ValueError("MSE eager search requires token_args")
        error_dtype = min_vals.dtype
        error_norm = norm

    best_error = torch.full(
        min_vals.shape,
        torch.finfo(error_dtype).max,
        device=min_vals.device,
        dtype=error_dtype,
    )
    best_min = min_vals.clone()
    best_max = max_vals.clone()
    no_improve_count = 0

    for i in range(int(maxshrink * grid)):
        p = 1 - i / grid
        shrink_min = min_vals * p
        shrink_max = max_vals * p
        if use_imatrix_error:
            error = _calculate_imatrix_error(
                observed,
                args,
                shrink_min,
                shrink_max,
                error_norm,
                error_dtype,
                importance_weights,
                total_values,
            )
        else:
            error = _calculate_mse_error(
                observed,
                args,
                token_args,
                shrink_min,
                shrink_max,
                norm,
            )

        improved = error < best_error
        if torch.any(improved):
            best_error[improved] = error[improved]
            best_min[improved] = shrink_min[improved]
            best_max[improved] = shrink_max[improved]
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience and (not use_imatrix_error or patience > 0):
                break

    return best_min, best_max


def _calculate_mse_error(
    observed: torch.Tensor,
    args: QuantizationArgs,
    token_args: QuantizationArgs,
    shrink_min: torch.Tensor,
    shrink_max: torch.Tensor,
    norm: float,
) -> torch.Tensor:
    """Calculate eager MSE error using the observer's established QDQ path."""
    scales, zero_points = calculate_qparams(
        min_vals=shrink_min,
        max_vals=shrink_max,
        quantization_args=args,
        global_scale=None,
    )
    quantized = fake_quantize(
        observed,
        scales.unsqueeze(-1),
        zero_points.unsqueeze(-1),
        token_args,
    ).to(observed.dtype)
    return (quantized - observed).abs().pow(norm).sum(dim=(0, -1))


def _calculate_imatrix_error(
    observed: torch.Tensor,
    args: QuantizationArgs,
    shrink_min: torch.Tensor,
    shrink_max: torch.Tensor,
    error_norm: float,
    error_dtype: torch.dtype,
    importance_weights: torch.Tensor | None,
    total_values: int,
) -> torch.Tensor:
    """Calculate imatrix error with Triton-matched rounding and reduction."""
    scales, zero_points = calculate_qparams(
        min_vals=shrink_min,
        max_vals=shrink_max,
        quantization_args=args,
        global_scale=None,
    )
    with patch_attr(args, "strategy", QuantizationStrategy.TOKEN):
        quantized = fake_quantize(
            observed,
            scales.unsqueeze(-1),
            zero_points.unsqueeze(-1),
            args,
        ).to(observed.dtype)

    error = (quantized - observed).abs().float().pow(error_norm)
    if error_dtype != torch.float32:
        error = error.to(error_dtype).float()
    if importance_weights is not None:
        error.mul_(importance_weights)

    if total_values <= 512:
        result = error.sum(dim=(0, -1))
    else:
        num_observations = observed.shape[0]
        num_qparams = math.prod(observed.shape[1:-1])
        group_size = observed.shape[-1]
        errors_by_qparam = (
            error.reshape(num_observations, num_qparams, group_size)
            .permute(1, 0, 2)
            .reshape(num_qparams, total_values)
        )
        partial_errors = [
            errors_by_qparam[:, offset : offset + 512].sum(dim=-1)
            for offset in range(0, total_values, 512)
        ]
        result = (
            torch.stack(partial_errors, dim=-1).sum(dim=-1).reshape(shrink_min.shape)
        )
    if error_dtype != torch.float32:
        result = result.to(error_dtype)
    return result


@ImplBackend.entrypoint("_grid_search_observer")
@torch.no_grad()
def _grid_search_observer(
    observed: torch.Tensor,
    args: QuantizationArgs,
    maxshrink: float,
    patience: int,
    grid: float,
    norm: float,
    triton_error_buffer: float,
    expand: float = 1.0,
    token_args: QuantizationArgs | None = None,
    importance_weights: torch.Tensor | None = None,
) -> MinMaxTuple:
    """Shared MSE/imatrix range-search entrypoint with optional weighting."""
    del triton_error_buffer
    if (
        args.strategy == QuantizationStrategy.TENSOR_GROUP
        and args.scale_dtype is not None
    ):
        args = args.model_copy(update={"scale_dtype": None})
        if token_args is not None:
            token_args = token_args.model_copy(update={"scale_dtype": None})

    return _grid_search_observer_eager(
        observed,
        args,
        maxshrink,
        patience,
        grid,
        norm,
        expand=expand,
        token_args=token_args,
        importance_weights=importance_weights,
        use_imatrix_error=token_args is None,
    )


@ImplBackend.register("_grid_search_observer", observer_triton_req, 0)
@torch.no_grad()
def _grid_search_observer_triton(
    observed: torch.Tensor,
    args: QuantizationArgs,
    maxshrink: float,
    patience: int,
    grid: float,
    norm: float,
    triton_error_buffer: float,
    expand: float = 1.0,
    token_args: QuantizationArgs | None = None,
    importance_weights: torch.Tensor | None = None,
) -> MinMaxTuple:
    """Shared Triton implementation of buffered MSE and imatrix search."""
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

    observed = observed.contiguous().reshape(observed.shape[0], -1, observed.shape[-1])
    num_observations, num_qparams, group_size = observed.shape
    has_importance = importance_weights is not None
    if importance_weights is not None:
        importance_weights = importance_weights.to(
            device=observed.device, dtype=torch.float32
        ).contiguous()
        if importance_weights.numel() != num_qparams * group_size:
            raise ValueError(
                "importance weights must contain one weight per qparam element: "
                f"expected {num_qparams * group_size}, got "
                f"{importance_weights.numel()}"
            )
        importance_weights = importance_weights.reshape(-1)
    else:
        # The constexpr guard removes all importance loads in the MSE specialization.
        importance_weights = observed
    total_steps = int(maxshrink * grid)
    if total_steps == 0:
        return min_val, max_val
    grid_points = torch.tensor(
        [1.0 - step / grid for step in range(total_steps)],
        device=observed.device,
        dtype=torch.float32,
    )
    best_step = torch.empty(num_qparams, dtype=torch.int32, device=observed.device)
    q_min, q_max = calculate_range(args, observed.device)
    scale_round_type, scale_eps = scale_round_config(args.scale_dtype)
    observed_dtype = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
    }[observed.dtype]
    quant_type = 0 if args.type == QuantizationType.INT else 1
    total_values = num_observations * group_size
    tile_values = 512
    if total_values <= tile_values:
        block_values = triton.next_power_of_2(total_values)
        tile_qparams = max(1, tile_values // block_values)
        _grid_search_observer_triton_packed_kernel[
            (triton.cdiv(num_qparams, tile_qparams),)
        ](
            observed,
            min_base,
            max_base,
            zp_base,
            importance_weights,
            best_step,
            num_observations,
            num_qparams,
            group_size,
            total_steps,
            grid_points,
            float(q_min),
            float(q_max),
            norm,
            patience,
            triton_error_buffer,
            scale_eps,
            BLOCK_VALUES=block_values,
            TILE_QPARAMS=tile_qparams,
            TOTAL_STEPS=triton.next_power_of_2(total_steps),
            QUANT_TYPE=quant_type,
            NUM_BITS=args.num_bits,
            HAS_ZP=not args.symmetric,
            HAS_IMPORTANCE=has_importance,
            SYMMETRIC=args.symmetric,
            SCALE_ROUND_TYPE=scale_round_type,
            OBSERVED_DTYPE=observed_dtype,
        )
    else:
        num_chunks = triton.cdiv(total_values, tile_values)
        partial_errors = torch.empty(
            total_steps,
            num_qparams,
            num_chunks,
            dtype=torch.float32,
            device=observed.device,
        )
        _grid_search_observer_triton_split_kernel[(num_qparams * num_chunks,)](
            observed,
            min_base,
            max_base,
            zp_base,
            importance_weights,
            partial_errors,
            num_observations,
            num_qparams,
            group_size,
            grid_points,
            float(q_min),
            float(q_max),
            norm,
            scale_eps,
            BLOCK_VALUES=tile_values,
            NUM_CHUNKS=num_chunks,
            TOTAL_STEPS=total_steps,
            QUANT_TYPE=quant_type,
            NUM_BITS=args.num_bits,
            HAS_ZP=not args.symmetric,
            HAS_IMPORTANCE=has_importance,
            SYMMETRIC=args.symmetric,
            SCALE_ROUND_TYPE=scale_round_type,
            OBSERVED_DTYPE=observed_dtype,
        )
        _grid_search_observer_triton_split_reduce_kernel[(num_qparams,)](
            partial_errors,
            best_step,
            num_qparams,
            patience,
            triton_error_buffer,
            BLOCK_CHUNKS=triton.next_power_of_2(num_chunks),
            NUM_CHUNKS=num_chunks,
            TOTAL_STEPS=total_steps,
        )
    best_p = grid_points[best_step.long()].reshape(min_val.shape)
    best_min = (min_val.to(torch.float32) * best_p).to(min_val.dtype)
    best_max = (max_val.to(torch.float32) * best_p).to(max_val.dtype)
    return best_min, best_max


@triton.jit
def _grid_search_observer_triton_packed_kernel(
    observed_ptr,
    min_base_ptr,
    max_base_ptr,
    zp_base_ptr,
    importance_ptr,
    best_step_ptr,
    num_observations,
    num_qparams,
    group_size,
    total_steps,
    grid_points_ptr,
    q_min,
    q_max,
    norm,
    patience,
    triton_error_buffer,
    scale_eps,
    BLOCK_VALUES: tl.constexpr,
    TILE_QPARAMS: tl.constexpr,
    TOTAL_STEPS: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    NUM_BITS: tl.constexpr,
    HAS_ZP: tl.constexpr,
    HAS_IMPORTANCE: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    SCALE_ROUND_TYPE: tl.constexpr,
    OBSERVED_DTYPE: tl.constexpr,
):
    """Pack complete qparams so each program handles about 512 values."""
    tile_offsets = tl.arange(0, TILE_QPARAMS)
    qparams = tl.program_id(0) * TILE_QPARAMS + tile_offsets
    qparam_mask = qparams < num_qparams
    value_offsets = tl.arange(0, BLOCK_VALUES)
    obs_idx = value_offsets // group_size
    group_idx = value_offsets % group_size
    value_mask = qparam_mask[:, None] & (
        value_offsets[None, :] < num_observations * group_size
    )
    values = tl.load(
        observed_ptr
        + obs_idx[None, :] * num_qparams * group_size
        + qparams[:, None] * group_size
        + group_idx[None, :],
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    min_base = tl.load(min_base_ptr + qparams, mask=qparam_mask, other=0.0).to(
        tl.float32
    )
    max_base = tl.load(max_base_ptr + qparams, mask=qparam_mask, other=0.0).to(
        tl.float32
    )
    zp = tl.load(zp_base_ptr + qparams, mask=qparam_mask, other=0.0).to(tl.float32)
    if HAS_IMPORTANCE:
        importance = tl.load(
            importance_ptr + qparams[:, None] * group_size + group_idx[None, :],
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        importance = 0.0
    if OBSERVED_DTYPE == 1:
        best_error = tl.full((TILE_QPARAMS,), float("inf"), tl.float16)
    elif OBSERVED_DTYPE == 2:
        best_error = tl.full((TILE_QPARAMS,), float("inf"), tl.bfloat16)
    else:
        best_error = tl.full((TILE_QPARAMS,), float("inf"), tl.float32)
    best_step = tl.zeros((TILE_QPARAMS,), tl.int32)
    stale = tl.zeros((TILE_QPARAMS,), tl.int32)

    for step in range(TOTAL_STEPS):
        if step < total_steps:
            active = stale < patience
            p = tl.load(grid_points_ptr + step)
            candidate_min = min_base * p
            candidate_max = max_base * p
            if OBSERVED_DTYPE == 1:
                candidate_min = candidate_min.to(tl.float16).to(tl.float32)
                candidate_max = candidate_max.to(tl.float16).to(tl.float32)
            elif OBSERVED_DTYPE == 2:
                candidate_min = candidate_min.to(tl.bfloat16).to(tl.float32)
                candidate_max = candidate_max.to(tl.bfloat16).to(tl.float32)
            scale = calculate_candidate_scale(
                candidate_min,
                candidate_max,
                1.0,
                q_min,
                q_max,
                SCALE_ROUND_TYPE=SCALE_ROUND_TYPE,
                QUANT_TYPE=QUANT_TYPE,
                SYMMETRIC=SYMMETRIC,
            )
            if SCALE_ROUND_TYPE == 0:
                if OBSERVED_DTYPE == 1:
                    scale = scale.to(tl.float16).to(tl.float32)
                elif OBSERVED_DTYPE == 2:
                    scale = scale.to(tl.bfloat16).to(tl.float32)
            scale = tl.maximum(scale, scale_eps)
            quantized = quantize_dequantize(
                values,
                scale[:, None],
                zp[:, None],
                q_min,
                q_max,
                QUANT_TYPE=QUANT_TYPE,
                NUM_BITS=NUM_BITS,
                HAS_ZP=HAS_ZP,
                COMPUTE_DTYPE=OBSERVED_DTYPE,
            )
            diff_pow = calculate_observer_error(
                values,
                quantized,
                importance,
                norm,
                HAS_IMPORTANCE=HAS_IMPORTANCE,
                COMPUTE_DTYPE=OBSERVED_DTYPE,
                ROUND_ERROR=True,
            )
            error = tl.sum(
                tl.where(value_mask, diff_pow, 0.0).to(tl.float32), axis=1
            ).to(best_error.dtype)
            previous_best = best_error
            is_better = active & (error < previous_best)
            best_error = tl.where(is_better, error, best_error)
            best_step = tl.where(is_better, step, best_step)
            within_buffer = error <= previous_best * (1.0 + triton_error_buffer)
            stale = tl.where(
                active & within_buffer,
                0,
                tl.where(active, stale + 1, stale),
            )

    tl.store(best_step_ptr + qparams, best_step, mask=qparam_mask)


@triton.jit
def _grid_search_observer_triton_split_kernel(
    observed_ptr,
    min_base_ptr,
    max_base_ptr,
    zp_base_ptr,
    importance_ptr,
    partial_error_ptr,
    num_observations,
    num_qparams,
    group_size,
    grid_points_ptr,
    q_min,
    q_max,
    norm,
    scale_eps,
    BLOCK_VALUES: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    TOTAL_STEPS: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    NUM_BITS: tl.constexpr,
    HAS_ZP: tl.constexpr,
    HAS_IMPORTANCE: tl.constexpr,
    SYMMETRIC: tl.constexpr,
    SCALE_ROUND_TYPE: tl.constexpr,
    OBSERVED_DTYPE: tl.constexpr,
):
    """Compute candidate partial errors for one 512-value qparam chunk."""
    pid = tl.program_id(0)
    qparam = pid // NUM_CHUNKS
    chunk = pid % NUM_CHUNKS
    qparam_mask = qparam < num_qparams
    flat_offsets = chunk * BLOCK_VALUES + tl.arange(0, BLOCK_VALUES)
    obs_idx = flat_offsets // group_size
    group_idx = flat_offsets % group_size
    value_mask = qparam_mask & (flat_offsets < num_observations * group_size)
    values = tl.load(
        observed_ptr
        + obs_idx * num_qparams * group_size
        + qparam * group_size
        + group_idx,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    min_base = tl.load(min_base_ptr + qparam, mask=qparam_mask, other=0.0).to(
        tl.float32
    )
    max_base = tl.load(max_base_ptr + qparam, mask=qparam_mask, other=0.0).to(
        tl.float32
    )
    zp = tl.load(zp_base_ptr + qparam, mask=qparam_mask, other=0.0).to(tl.float32)
    if HAS_IMPORTANCE:
        importance = tl.load(
            importance_ptr + qparam * group_size + group_idx,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        importance = 0.0

    for step in tl.static_range(0, TOTAL_STEPS):
        scale = calculate_candidate_scale(
            min_base,
            max_base,
            tl.load(grid_points_ptr + step),
            q_min,
            q_max,
            SCALE_ROUND_TYPE=SCALE_ROUND_TYPE,
            QUANT_TYPE=QUANT_TYPE,
            SYMMETRIC=SYMMETRIC,
        )
        scale = tl.maximum(scale, scale_eps)
        quantized = quantize_dequantize(
            values,
            scale,
            zp,
            q_min,
            q_max,
            QUANT_TYPE=QUANT_TYPE,
            NUM_BITS=NUM_BITS,
            HAS_ZP=HAS_ZP,
            COMPUTE_DTYPE=OBSERVED_DTYPE,
        )
        diff_pow = calculate_observer_error(
            values,
            quantized,
            importance,
            norm,
            HAS_IMPORTANCE=HAS_IMPORTANCE,
            COMPUTE_DTYPE=OBSERVED_DTYPE,
            ROUND_ERROR=False,
        )
        error = tl.sum(tl.where(value_mask, diff_pow, 0.0))
        tl.store(
            partial_error_ptr + (step * num_qparams + qparam) * NUM_CHUNKS + chunk,
            error,
            mask=qparam_mask,
        )


@triton.jit
def _grid_search_observer_triton_split_reduce_kernel(
    partial_error_ptr,
    best_step_ptr,
    num_qparams,
    patience,
    triton_error_buffer,
    BLOCK_CHUNKS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    TOTAL_STEPS: tl.constexpr,
):
    """Reduce chunk errors and apply buffered patience per qparam."""
    qparam = tl.program_id(0)
    qparam_mask = qparam < num_qparams
    chunks = tl.arange(0, BLOCK_CHUNKS)
    chunk_mask = chunks < NUM_CHUNKS
    best_error = tl.full([], float("inf"), tl.float32)
    best_step = 0
    stale = 0
    for step in tl.static_range(0, TOTAL_STEPS):
        partials = tl.load(
            partial_error_ptr + (step * num_qparams + qparam) * NUM_CHUNKS + chunks,
            mask=qparam_mask & chunk_mask,
            other=0.0,
        )
        error = tl.sum(partials)
        active = stale < patience
        previous_best = best_error
        is_better = active & (error < previous_best)
        best_error = tl.where(is_better, error, best_error)
        best_step = tl.where(is_better, step, best_step)
        within_buffer = error <= previous_best * (1.0 + triton_error_buffer)
        stale = tl.where(active & within_buffer, 0, tl.where(active, stale + 1, stale))
    tl.store(best_step_ptr + qparam, best_step, mask=qparam_mask)
