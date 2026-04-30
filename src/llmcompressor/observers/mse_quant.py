from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._dynamo.decorators
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam

from llmcompressor.observers.base import MinMaxTuple
from llmcompressor.observers.compile_config import (
    get_compile_chunk_size,
    get_torch_compile,
)

# Allow torch.compile to handle scalar conversions inside
# compressed_tensors' calculate_qparams (float(bit_range)).
# Same approach as GPTQ compile path (commit a4f9ba2e).
torch._dynamo.config.capture_scalar_outputs = True


def _grid_search_mse(
    observed: torch.Tensor,
    args: QuantizationArgs,
    token_args: QuantizationArgs,
    maxshrink: float,
    patience: int,
    grid: float,
    norm: float,
    global_scale: Optional[torch.Tensor] = None,
    optimize_global_scale: bool = False,
) -> MinMaxTuple:
    """Find per-channel min/max ranges that minimize quantization error.

    Performs a 1-D grid search over shrink factors applied to the observed
    tensor's min/max values.  Delegates to :func:`_grid_search_eager` or
    :func:`_grid_search_compiled` based on the global compile config set
    via :func:`set_torch_compile`.

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
    :param global_scale: precomputed global scale to use for quantization. Ignored if
        `optimize_global_scale` is True
    :param optimize_global_scale: If True, recompute ``global_scale`` from the
        candidate min/max during each step of the search
    """
    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    best_error = torch.full_like(min_val, torch.finfo(min_val.dtype).max)
    best_min_val = min_val.clone()
    best_max_val = max_val.clone()

    total_steps = int(maxshrink * grid)

    dispatch = _grid_search_compiled if get_torch_compile() else _grid_search_eager
    return dispatch(
        observed,
        args,
        token_args,
        min_val,
        max_val,
        best_error,
        best_min_val,
        best_max_val,
        total_steps,
        patience,
        grid,
        norm,
        global_scale,
        optimize_global_scale,
    )


def _grid_search_eager(
    observed: torch.Tensor,
    args: QuantizationArgs,
    token_args: QuantizationArgs,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    best_error: torch.Tensor,
    best_min_val: torch.Tensor,
    best_max_val: torch.Tensor,
    total_steps: int,
    patience: int,
    grid: float,
    norm: float,
    global_scale: Optional[torch.Tensor],
    optimize_global_scale: bool,
) -> MinMaxTuple:
    """Per-step grid search with boolean-indexing updates and early stopping."""
    no_improve_count = 0

    # @ksayers @HGCharles: investigate searching over separate min/max
    for i in range(total_steps):
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
            global_scale,
            optimize_global_scale,
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


def _grid_search_compiled(
    observed: torch.Tensor,
    args: QuantizationArgs,
    token_args: QuantizationArgs,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    best_error: torch.Tensor,
    best_min_val: torch.Tensor,
    best_max_val: torch.Tensor,
    total_steps: int,
    patience: int,
    grid: float,
    norm: float,
    global_scale: Optional[torch.Tensor],
    optimize_global_scale: bool,
) -> MinMaxTuple:
    """Chunked grid search using torch.compiled inner loop.

    Batches ``chunk_size`` candidates per compiled call to amortise
    compile/dispatch overhead. Early stopping is checked at chunk
    boundaries; compiled mode may run up to ``chunk_size - 1`` extra
    steps past the eager break point.
    """
    chunk_size = get_compile_chunk_size()
    no_improve_count = 0

    # Eliminate stride/duck-sizing guards from view tensors
    observed = observed.clone()
    # Prevent size specialization on group-size dim
    torch._dynamo.decorators.mark_unbacked(observed, observed.ndim - 1)

    idx = 0
    while idx < total_steps:
        chunk_end = min(idx + chunk_size, total_steps)
        current_chunk = chunk_end - idx

        ps = torch.tensor(
            [1.0 - (idx + j) / grid for j in range(current_chunk)],
            dtype=observed.dtype,
            device=observed.device,
        )

        prev_best = best_error.clone()
        best_error, best_min_val, best_max_val = _compute_chunk(
            observed,
            args,
            token_args,
            min_val,
            max_val,
            ps,
            current_chunk,
            norm,
            global_scale,
            optimize_global_scale,
            best_error,
            best_min_val,
            best_max_val,
        )

        if torch.equal(prev_best, best_error):
            no_improve_count += current_chunk
            if no_improve_count >= patience:
                break
        else:
            no_improve_count = 0
        idx = chunk_end

    return best_min_val, best_max_val


@torch.compile(dynamic=True)
def _compute_chunk(
    observed: torch.Tensor,
    args: QuantizationArgs,
    token_args: QuantizationArgs,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    ps: torch.Tensor,
    chunk_size: int,
    norm: float,
    global_scale: Optional[torch.Tensor],
    optimize_global_scale: bool,
    best_error: torch.Tensor,
    best_min_val: torch.Tensor,
    best_max_val: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate ``chunk_size`` shrink-factor candidates in one call.

    Uses ``torch.where`` to update best results so that each candidate's
    full fake-quantized tensor can be freed before the next iteration.

    When compiled, dynamo unrolls the inner loop (``chunk_size`` is a
    Python int constant) and fuses the resulting ops into fewer kernels.

    :return: (best_error, best_min_val, best_max_val) updated
    """
    for j in range(chunk_size):
        shrinked_min = min_val * ps[j]
        shrinked_max = max_val * ps[j]

        err = _calculate_error(
            observed,
            args,
            token_args,
            shrinked_min,
            shrinked_max,
            norm,
            global_scale,
            optimize_global_scale,
        )

        improved = err < best_error
        best_error = torch.where(improved, err, best_error)
        best_min_val = torch.where(improved, shrinked_min, best_min_val)
        best_max_val = torch.where(improved, shrinked_max, best_max_val)

    return best_error, best_min_val, best_max_val


def _calculate_error(
    observed: torch.Tensor,
    args: QuantizationArgs,
    token_args: QuantizationArgs,
    shrinked_min: torch.Tensor,
    shrinked_max: torch.Tensor,
    norm: float,
    global_scale: Optional[torch.Tensor],
    optimize_global_scale: bool,
) -> torch.Tensor:
    """Fake-quantize ``observed`` using the given shrinked min/max range and
    return the per-channel error.

    :return: per-channel quantization error, shape ``(*qparams_shape,)``
    """
    gs = global_scale
    if optimize_global_scale:
        gs = generate_gparam(shrinked_min, shrinked_max)

    candidate_scales, candidate_zero_points = calculate_qparams(
        min_vals=shrinked_min,
        max_vals=shrinked_max,
        quantization_args=args,
        global_scale=gs,
    )

    q = fake_quantize(
        observed,
        candidate_scales.unsqueeze(-1),
        candidate_zero_points.unsqueeze(-1),
        token_args,
        global_scale=gs,
    ).to(observed.dtype)

    err = torch.sum((q - observed).abs().pow(norm), dim=(0, -1))
    del q
    return err
