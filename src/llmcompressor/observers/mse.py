from typing import Optional, Tuple

import torch
import torch._dynamo.config
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.compile_config import (
    get_compile_chunk_size,
    get_torch_compile,
)
from llmcompressor.observers.moving_base import MovingAverageObserverBase

__all__ = ["MovingAverageMSEObserver"]

# Allow torch.compile to handle scalar conversions inside
# compressed_tensors' calculate_qparams (float(bit_range)).
# Same approach as GPTQ compile path (commit a4f9ba2e).
torch._dynamo.config.capture_scalar_outputs = True


@Observer.register("memoryless_mse")
class MemorylessMSEObserver(Observer):
    """
    Compute quantization parameters by finding the optimal min/max values which minimize
    the mean of quantization error squared

    ```psuedocode
    mse_quant_error := mean((x - fake_quant(x))**2)
    global_scale <- min[min_vals, max_vals, global_scale](mse_quant_error(x))
    scale, zp <- min[min_vals, max_vals](mse_quant_error(x, global_scale))
    ```

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization\n
        maxshrink: maximum shrink amount (in "grid steps"). The number of
            search steps is int(maxshrink * grid)\n
        patience: number of consecutive search steps without improvement before
            early stopping\n
        grid: resolution of the shrink search. Larger values give finer granularity
            in shrink factors\n
        norm: exponent used when computing the error. norm = 2 approximates MSE\n
        global_scale: precomputed global scale to use for quantization. Ignored if
            `optimize_global_scale` is True\n
        optimize_global_scale: If True, recompute ``global_scale`` from the
            candidate min/max during each step of the search
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.patience = observer_kwargs.get("patience", 5)
        self.grid = observer_kwargs.get("grid", 100.0)
        self.norm = observer_kwargs.get("norm", 2.4)

        # Pre-create token_args to avoid patch_attr context manager
        # which causes torch.compile graph breaks
        self._token_args = self.args.model_copy(
            update={"strategy": QuantizationStrategy.TOKEN}
        )

    def _call_grid_search(
        self,
        observed: torch.Tensor,
        global_scale: Optional[torch.Tensor],
        optimize_global_scale: bool,
    ) -> MinMaxTuple:
        return _grid_search_mse(
            observed,
            self.args,
            self._token_args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            global_scale=global_scale,
            optimize_global_scale=optimize_global_scale,
            enable_compile=get_torch_compile(),
            chunk_size=get_compile_chunk_size(),
        )

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        # min[min_vals, max_vals](mse_quant_error)
        global_scale = self._get_module_param("global_scale")
        return self._call_grid_search(observed, global_scale, False)

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        # min[min_vals, max_vals, global_scale](mse_quant_error)
        return self._call_grid_search(observed, None, True)


@Observer.register("mse")
class MovingAverageMSEObserver(MovingAverageObserverBase):
    """
    Compute quantization parameters by finding the optimal min/max values which minimize
    the mean of quantization error squared.

    ```psuedocode
    mse_quant_error := mean((x - fake_quant(x))**2)
    global_scale <- min[min_vals, max_vals, global_scale](mse_quant_error(x))
    scale, zp <- min[min_vals, max_vals](mse_quant_error(x, global_scale))
    ```

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization\n
        maxshrink: maximum shrink amount (in "grid steps"). The number of
            search steps is int(maxshrink * grid)\n
        patience: number of consecutive search steps without improvement before
            early stopping\n
        grid: resolution of the shrink search. Larger values give finer granularity
            in shrink factors\n
        norm: exponent used when computing the error. norm = 2 approximates MSE\n
        global_scale: precomputed global scale to use for quantization. Ignored if
            `optimize_global_scale` is True\n
        optimize_global_scale: If True, recompute ``global_scale`` from the
            candidate min/max during each step of the search
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.patience = observer_kwargs.get("patience", 5)
        self.grid = observer_kwargs.get("grid", 100.0)
        self.norm = observer_kwargs.get("norm", 2.4)

        # Pre-create token_args to avoid patch_attr context manager
        # which causes torch.compile graph breaks
        self._token_args = self.args.model_copy(
            update={"strategy": QuantizationStrategy.TOKEN}
        )

    def _call_grid_search(
        self,
        observed: torch.Tensor,
        global_scale: Optional[torch.Tensor],
        optimize_global_scale: bool,
    ) -> MinMaxTuple:
        return _grid_search_mse(
            observed,
            self.args,
            self._token_args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            global_scale=global_scale,
            optimize_global_scale=optimize_global_scale,
            enable_compile=get_torch_compile(),
            chunk_size=get_compile_chunk_size(),
        )

    def get_current_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        # min[min_vals, max_vals](mse_quant_error)
        global_scale = self._get_module_param("global_scale")
        return self._call_grid_search(observed, global_scale, False)

    def get_current_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        # min[min_vals, max_vals, global_scale](mse_quant_error)
        return self._call_grid_search(observed, None, True)


def _compute_candidate_error(
    observed: torch.Tensor,
    args: QuantizationArgs,
    token_args: QuantizationArgs,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    p: float,
    norm: float,
    global_scale: Optional[torch.Tensor],
    optimize_global_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the quantization error for a single shrink factor.

    Shared helper used by the grid search. When enable_compile is set
    via oneshot, this function is called through its compiled wrapper
    for accelerated execution.

    :param observed: value of shape (num_observations, *qparams_shape, group_size)
    :param args: quantization args used for computing qparams
    :param token_args: quantization args with strategy set to TOKEN, pre-created
        to avoid patch_attr context manager which causes torch.compile graph breaks
    :param min_val: per-channel minimum values
    :param max_val: per-channel maximum values
    :param p: shrink factor (1 - i/grid)
    :param norm: exponent used when computing the error
    :param global_scale: precomputed global scale to use for quantization
    :param optimize_global_scale: If True, recompute global_scale from candidates
    :return: (error, shrinked_min_val, shrinked_max_val)
    """
    shrinked_min_val = p * min_val
    shrinked_max_val = p * max_val

    if optimize_global_scale:
        global_scale = generate_gparam(shrinked_min_val, shrinked_max_val)

    candidate_scales, candidate_zero_points = calculate_qparams(
        min_vals=shrinked_min_val,
        max_vals=shrinked_max_val,
        quantization_args=args,
        global_scale=global_scale,
    )

    # Use pre-created token_args instead of patch_attr context manager
    # to maintain torch.compile compatibility
    q = fake_quantize(
        observed,
        candidate_scales.unsqueeze(-1),
        candidate_zero_points.unsqueeze(-1),
        token_args,
        global_scale=global_scale,
    ).to(observed.dtype)

    err = torch.sum((q - observed).abs().pow(norm), dim=(0, -1))
    return err, shrinked_min_val, shrinked_max_val


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
    """
    Compute quantization errors for multiple shrink factors in one call,
    updating best results via torch.where so that each candidate's full
    fake_quantize output can be freed before the next iteration.

    When compiled, torch.compile unrolls the inner loop (chunk_size is a
    Python int constant from dynamo's perspective) and fuses the resulting
    operations into fewer, larger kernels.  Because only the scalar error
    and qparams are carried forward (not the full quantized tensor), peak
    memory stays constant regardless of chunk_size.

    :param observed: value of shape (num_observations, *qparams_shape, group_size)
    :param args: quantization args used for computing qparams
    :param token_args: quantization args with strategy set to TOKEN
    :param min_val: per-channel minimum values
    :param max_val: per-channel maximum values
    :param ps: 1-D tensor of shrink factors, length == chunk_size
    :param chunk_size: number of candidates in this chunk (Python int so
        dynamo can unroll the loop at trace time)
    :param norm: exponent used when computing the error
    :param global_scale: precomputed global scale to use for quantization
    :param optimize_global_scale: If True, recompute global_scale from candidates
    :param best_error: current best per-channel errors
    :param best_min_val: current best per-channel min values
    :param best_max_val: current best per-channel max values
    :return: (best_error, best_min_val, best_max_val) updated
    """
    for j in range(chunk_size):
        p = ps[j]
        shrinked_min = p * min_val
        shrinked_max = p * max_val

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

        improved = err < best_error
        best_error = torch.where(improved, err, best_error)
        best_min_val = torch.where(improved, shrinked_min, best_min_val)
        best_max_val = torch.where(improved, shrinked_max, best_max_val)

    return best_error, best_min_val, best_max_val


# Compiled variants.
# _compute_chunk_compiled: processes chunk_size candidates per call,
# reducing per-call overhead by the chunk factor.
_compute_chunk_compiled = torch.compile(_compute_chunk, dynamic=True)


def _grid_search_mse(
    observed: torch.Tensor,
    args: QuantizationArgs,
    token_args: QuantizationArgs,
    maxshrink: float,
    patience: float,
    grid: float,
    norm: float,
    global_scale: Optional[torch.Tensor] = None,
    optimize_global_scale: bool = False,
    enable_compile: bool = False,
    chunk_size: int = 5,
) -> MinMaxTuple:
    """
    Perform a 1-D grid search to find per-channel min/max ranges that minimize
    mean-squared quantization error.

    Progressively shrinks the absolute min/max ranges of the observed tensor
    and evaluates the quantization error at each candidate. Early stopping
    exits when no improvement is found for ``patience`` consecutive steps.

    When enable_compile is True and chunk_size > 1, multiple grid steps are
    batched into a single torch.compiled call.  The compiled function unrolls
    the inner loop so that chunk_size candidates share one kernel launch,
    amortising compile/dispatch overhead.  Early stopping is checked between
    chunks to preserve convergence behaviour.

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
    :param enable_compile: If True, use torch.compiled inner computation
    :param chunk_size: number of grid steps per compiled call (default 5).
        Ignored when enable_compile is False.
    """
    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    best_error = torch.full_like(min_val, torch.finfo(min_val.dtype).max)
    best_min_val = min_val.clone()
    best_max_val = max_val.clone()

    total_steps = int(maxshrink * grid)
    no_improve_count = 0

    if enable_compile:
        # [recompile fix] Eliminate stride/duck-sizing guards from view tensors
        observed = observed.clone()

        # [recompile fix] Prevent size specialization on group-size dim
        torch._dynamo.decorators.mark_unbacked(observed, observed.ndim - 1)

        # Chunked compile path: batch multiple candidates per compiled call.
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
            best_error, best_min_val, best_max_val = _compute_chunk_compiled(
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
            # Chunk-level early stopping: if no channel improved across
            # the entire chunk, increment by chunk size towards patience.
            if torch.equal(prev_best, best_error):
                no_improve_count += current_chunk
                if no_improve_count >= patience:
                    break
            else:
                no_improve_count = 0
            idx = chunk_end
    else:
        # Eager path.
        compute_fn = _compute_candidate_error

        # @ksayers @HGCharles: investigate searching over separate shrinking
        # factors
        for i in range(total_steps):
            p = 1 - i / grid
            err, shrinked_min_val, shrinked_max_val = compute_fn(
                observed,
                args,
                token_args,
                min_val,
                max_val,
                p,
                norm,
                global_scale,
                optimize_global_scale,
            )

            tmp = err < best_error
            if torch.any(tmp):
                best_error[tmp] = err[tmp]
                best_min_val[tmp] = shrinked_min_val[tmp]
                best_max_val[tmp] = shrinked_max_val[tmp]
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    break

    return best_min_val, best_max_val
