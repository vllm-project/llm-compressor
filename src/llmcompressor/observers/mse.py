from typing import Optional, Tuple

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.moving_base import MovingAverageObserverBase

__all__ = ["MovingAverageMSEObserver"]


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
            early stopping. Only used when enable_torch_compile is False\n
        grid: resolution of the shrink search. Larger values give finer granularity
            in shrink factors\n
        norm: exponent used when computing the error. norm = 2 approximates MSE\n
        global_scale: precomputed global scale to use for quantization. Ignored if
            `optimize_global_scale` is True\n
        optimize_global_scale: If True, recompute ``global_scale`` from the
            candidate min/max during each step of the search\n
        enable_torch_compile: If True, use a torch.compile-compatible code path
            that removes early stopping and patch_attr context manager. Default False
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.patience = observer_kwargs.get("patience", 5)
        self.grid = observer_kwargs.get("grid", 100.0)
        self.norm = observer_kwargs.get("norm", 2.4)
        self.enable_torch_compile = observer_kwargs.get("enable_torch_compile", False)

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
        if self.enable_torch_compile:
            return _grid_search_mse_compiled(
                observed,
                self.args,
                self._token_args,
                self.maxshrink,
                self.grid,
                self.norm,
                global_scale=global_scale,
                optimize_global_scale=optimize_global_scale,
            )
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
            early stopping. Only used when enable_torch_compile is False\n
        grid: resolution of the shrink search. Larger values give finer granularity
            in shrink factors\n
        norm: exponent used when computing the error. norm = 2 approximates MSE\n
        global_scale: precomputed global scale to use for quantization. Ignored if
            `optimize_global_scale` is True\n
        optimize_global_scale: If True, recompute ``global_scale`` from the
            candidate min/max during each step of the search\n
        enable_torch_compile: If True, use a torch.compile-compatible code path
            that removes early stopping and patch_attr context manager. Default False
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.patience = observer_kwargs.get("patience", 5)
        self.grid = observer_kwargs.get("grid", 100.0)
        self.norm = observer_kwargs.get("norm", 2.4)
        self.enable_torch_compile = observer_kwargs.get("enable_torch_compile", False)

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
        if self.enable_torch_compile:
            return _grid_search_mse_compiled(
                observed,
                self.args,
                self._token_args,
                self.maxshrink,
                self.grid,
                self.norm,
                global_scale=global_scale,
                optimize_global_scale=optimize_global_scale,
            )
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

    Shared helper used by both the default and torch.compile-compatible
    grid search paths to avoid code duplication.

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
    # Note that due to forward quantization implementation, token quant,
    # unlike tensor_group, requires extra dtype cast

    err = torch.sum((q - observed).abs().pow(norm), dim=(0, -1))
    return err, shrinked_min_val, shrinked_max_val


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
) -> MinMaxTuple:
    """
    Perform a 1-D grid search to find per-channel min/max ranges that minimize
    mean-squared quantization error.

    This routine progressively "shrinks" the absolute min/max ranges of the
    observed tensor and evaluates the quantization error at each candidate
    range. For each shrink factor ``p = 1 - i/grid`` up to ``maxshrink``.

    Uses early stopping to skip unnecessary search steps when no improvement
    is found for ``patience`` consecutive steps.

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

    no_improve_count = 0

    # @ksayers @HGCharles: investigate searching over separate shrinking factors
    for i in range(int(maxshrink * grid)):
        p = 1 - i / grid
        err, shrinked_min_val, shrinked_max_val = _compute_candidate_error(
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


def _grid_search_mse_compiled(
    observed: torch.Tensor,
    args: QuantizationArgs,
    token_args: QuantizationArgs,
    maxshrink: float,
    grid: float,
    norm: float,
    global_scale: Optional[torch.Tensor] = None,
    optimize_global_scale: bool = False,
) -> MinMaxTuple:
    """
    torch.compile-compatible version of _grid_search_mse.

    Differences from the default path:
    - Uses torch.where instead of data-dependent control flow
      (early stopping and torch.any cause graph breaks)
    - No early stopping: runs all search steps for deterministic compilation

    :param observed: value of shape (num_observations, *qparams_shape, group_size)
    :param args: quantization args used for computing qparams
    :param token_args: quantization args with strategy set to TOKEN
    :param maxshrink: maximum shrink amount. The number of search steps is
        int(maxshrink * grid)
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

    num_steps = int(maxshrink * grid)
    for i in range(num_steps):
        p = 1 - i / grid
        err, shrinked_min_val, shrinked_max_val = _compute_candidate_error(
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

        # Use torch.where instead of boolean indexing + torch.any for
        # torch.compile compatibility (avoids data-dependent control flow)
        improved = err < best_error
        best_error = torch.where(improved, err, best_error)
        best_min_val = torch.where(improved, shrinked_min_val, best_min_val)
        best_max_val = torch.where(improved, shrinked_max_val, best_max_val)

    return best_min_val, best_max_val
