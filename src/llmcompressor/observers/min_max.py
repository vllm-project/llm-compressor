import torch
from torch import distributed as dist

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.moving_base import MovingAverageObserverBase

__all__ = ["MemorylessMinMaxObserver", "StaticMinMaxObserver", "MinMaxObserver"]


@Observer.register("memoryless_minmax")
class MemorylessMinMaxObserver(Observer):
    """
    Compute quantization parameters by taking the min/max of the observed value.
    """

    _act_sync_dict = {}

    def update_statistics(self, observed: torch.Tensor) -> None:
        self.min_vals, self.max_vals = _get_min_max(observed)


@Observer.register("static_minmax")
class StaticMinMaxObserver(MemorylessMinMaxObserver):
    """
    Compute quantization parameters by taking the min/max of all observed values.
    """

    _act_sync_dict = {
        "min_vals": dist.ReduceOp.MIN,
        "max_vals": dist.ReduceOp.MAX,
    }

    def update_statistics(self, observed: torch.Tensor) -> None:
        min_vals, max_vals = _get_min_max(observed)

        if hasattr(self, "min_vals"):
            self.min_vals = torch.min(min_vals, self.min_vals)
            self.max_vals = torch.max(max_vals, self.max_vals)
        else:
            self.min_vals = min_vals
            self.max_vals = max_vals


@Observer.register("minmax")
class MinMaxObserver(MovingAverageObserverBase):
    """
    Compute quantization parameters by taking the moving average of min/max values.
    """

    def update_statistics(self, observed: torch.Tensor) -> None:
        min_vals, max_vals = _get_min_max(observed)

        if hasattr(self, "min_vals") and self.avg_constant != 1.0:
            min_vals = self._lerp(self.min_vals, min_vals, self.avg_constant)
            max_vals = self._lerp(self.max_vals, max_vals, self.avg_constant)

        self.min_vals = min_vals
        self.max_vals = max_vals


def _get_min_max(observed: torch.Tensor) -> MinMaxTuple:
    min_vals = torch.amin(observed, dim=(0, -1))
    max_vals = torch.amax(observed, dim=(0, -1))

    return min_vals, max_vals
