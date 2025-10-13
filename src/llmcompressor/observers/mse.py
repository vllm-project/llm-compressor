from typing import Optional

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.utils import patch_attr

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.moving_base import MovingAverageObserverBase

__all__ = ["MovingAverageMSEObserver"]


@Observer.register("memoryless_mse")
class MemorylessMSEObserver(Observer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.patience = observer_kwargs.get("patience", 5)
        self.grid = observer_kwargs.get("grid", 100.0)
        self.norm = observer_kwargs.get("norm", 2.4)

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        global_scale = self._get_module_param("global_scale")
        return _grid_search_mse(
            observed,
            self.args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            global_scale=global_scale,
            optimize_global_scale=False,
        )

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _grid_search_mse(
            observed,
            self.args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            global_scale=None,
            optimize_global_scale=True,
        )


@Observer.register("mse")
class MovingAverageMSEObserver(MovingAverageObserverBase):
    """
    Implements a dynamic quantization observer that sets the scale and
    zero point based on a moving average of the mse-clipped min and max observed values
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.patience = observer_kwargs.get("patience", 5)
        self.grid = observer_kwargs.get("grid", 100.0)
        self.norm = observer_kwargs.get("norm", 2.4)

    def get_current_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        global_scale = self._get_module_param("global_scale")
        return _grid_search_mse(
            observed,
            self.args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            global_scale=global_scale,
            optimize_global_scale=False,
        )

    def get_current_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _grid_search_mse(
            observed,
            self.args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            global_scale=None,
            optimize_global_scale=True,
        )


def _grid_search_mse(
    observed: torch.Tensor,
    args: QuantizationArgs,
    maxshrink: float,
    patience: float,
    grid: float,
    norm: float,
    global_scale: Optional[torch.Tensor] = None,
    optimize_global_scale: bool = False,
) -> MinMaxTuple:
    """
    Grid search for MSE-optimal min and max values. If global scale is not given,
    then

    :param observed: value being observed whose shape is
        (num_observations, *qparam_shape, group_size)
    :return: minimum and maximum values which minimize reconstruction error
    """
    absolute_min_val = torch.amin(observed, dim=(0, -1))
    absolute_max_val = torch.amax(observed, dim=(0, -1))
    best = torch.full_like(absolute_min_val, torch.finfo(absolute_min_val.dtype).max)
    min_val = torch.ones_like(absolute_min_val)
    max_val = torch.zeros_like(absolute_max_val)

    # Early stopping params
    no_improve_count = 0

    # @ksayers @HGCharles: investigate searching over separate shrinking factors
    for i in range(int(maxshrink * grid)):
        p = 1 - i / grid
        shrinked_min_val = p * absolute_min_val
        shrinked_max_val = p * absolute_max_val

        if optimize_global_scale:
            global_scale = generate_gparam(shrinked_min_val, shrinked_max_val)

        candidate_scales, candidate_zero_points = calculate_qparams(
            min_vals=shrinked_min_val,
            max_vals=shrinked_max_val,
            quantization_args=args,
            global_scale=global_scale,
        )

        # Note that observed.shape = (num_observations, *qparams_shape, group_size).
        # For the purposes of fake quantization, this is equivalent to token quant
        with patch_attr(args, "strategy", QuantizationStrategy.TOKEN):
            q = fake_quantize(
                observed,
                candidate_scales.unsqueeze(-1),
                candidate_zero_points.unsqueeze(-1),
                args,
                global_scale=global_scale,
            ).to(observed.dtype)
            # Note that due to forward quantization implementation, token quant,
            # unlike tensor_group, requires extra dtype cast

        q -= observed
        q.abs_()
        q.pow_(norm)
        err = torch.sum(q, dim=(0, -1))

        tmp = err < best
        if torch.any(tmp):
            best[tmp] = err[tmp]
            min_val[tmp] = shrinked_min_val[tmp]
            max_val[tmp] = shrinked_max_val[tmp]
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

    return min_val, max_val
