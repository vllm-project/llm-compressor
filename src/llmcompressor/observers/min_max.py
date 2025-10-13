import torch

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.moving_base import MovingAverageObserverBase

__all__ = ["MemorylessMinMaxObserver", "StaticMinMaxObserver", "MinMaxObserver"]


@Observer.register("memoryless_minmax")
class MemorylessMinMaxObserver(Observer):
    """
    TODO
    """

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_min_max(observed)

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_min_max(observed)


@Observer.register("static_minmax")
class StaticMinMaxObserver(Observer):
    """
    TODO
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.past_min_vals = None
        self.past_max_vals = None
        self.past_global_min_vals = None
        self.past_global_max_vals = None

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        min_vals, max_vals = _get_min_max(observed)

        if self.past_min_vals is not None:
            min_vals = torch.min(min_vals, self.past_min_vals)
            max_vals = torch.max(max_vals, self.past_max_vals)

        self.past_min_vals = min_vals
        self.past_max_vals = max_vals

        return min_vals, max_vals

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        min_vals, max_vals = _get_min_max(observed)

        if self.past_global_min_vals is not None:
            min_vals = torch.min(min_vals, self.past_global_min_vals)
            max_vals = torch.max(max_vals, self.past_global_max_vals)

        self.past_global_min_vals = min_vals
        self.past_global_max_vals = max_vals

        return min_vals, max_vals


@Observer.register("minmax")
class MinMaxObserver(MovingAverageObserverBase):
    """
    TODO
    """

    def get_current_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_min_max(observed)

    def get_current_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_min_max(observed)


def _get_min_max(observed: torch.Tensor) -> MinMaxTuple:
    min_vals = torch.amin(observed, dim=(0, -1))
    max_vals = torch.amax(observed, dim=(0, -1))

    return min_vals, max_vals
