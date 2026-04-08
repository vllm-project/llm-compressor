from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.compile_config import (
    get_compile_chunk_size,
    get_torch_compile,
)
from llmcompressor.observers.moving_base import MovingAverageObserverBase
from llmcompressor.observers.mse_quant import _grid_search_mse  # noqa: E402

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


