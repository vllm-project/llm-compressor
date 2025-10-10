from typing import Tuple

import torch

from llmcompressor.observers.base import Observer
from llmcompressor.observers.moving_base import MovingAverageObserverBase
from llmcompressor.observers.static_base import StaticObserverBase

__all__ = ["StaticMinMaxObserver", "MinMaxObserver"]


@Observer.register("static_minmax")
class StaticMinMaxObserver(StaticObserverBase):
    """
    Implements a quantization observer that calculates scale and zero point based on the
    the minimum and maximum values of all observed values
    """

    def get_current_min_max(
        self, observed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        min_vals = torch.amin(observed, dim=(0, -1))
        max_vals = torch.amax(observed, dim=(0, -1))

        return min_vals, max_vals


@Observer.register("minmax")
class MinMaxObserver(MovingAverageObserverBase):
    """
    Implements a quantization observer that calculates scale and zero point based on the
    minimum and maximum values of the tensor being observed. If averaging_constant is
    specified, then the scales are updated using a moving average
    """

    def get_current_min_max(
        self, observed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return StaticMinMaxObserver.get_current_min_max(self, observed)
