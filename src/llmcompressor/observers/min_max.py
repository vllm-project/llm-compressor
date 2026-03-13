from typing import Optional

import torch

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.moving_base import MovingAverageObserverBase

__all__ = ["MemorylessMinMaxObserver", "StaticMinMaxObserver", "MinMaxObserver"]


@Observer.register("memoryless_minmax")
class MemorylessMinMaxObserver(Observer):
    """
    Compute quantization parameters by taking the min/max of the observed value

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_min_max(observed)

    def calculate_qparams(
        self, global_scale: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """
        Calculate quantization parameters and clear accumulated min/max values.

        :param global_scale: optional pre-computed global scale for tensor-group quantization
        :return: dictionary mapping parameter names to their computed values
        """
        ret = super().calculate_qparams(global_scale)
        self.min_vals = None
        self.max_vals = None

        return ret


@Observer.register("static_minmax")
class StaticMinMaxObserver(Observer):
    """
    Compute quantization parameters by taking the min/max of all observed values

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        min_vals, max_vals = _get_min_max(observed)

        if self.min_vals is not None:
            min_vals = torch.min(min_vals, self.min_vals)
        if self.max_vals is not None:
            max_vals = torch.max(max_vals, self.max_vals)

        return min_vals, max_vals


@Observer.register("minmax")
class MinMaxObserver(MovingAverageObserverBase):
    """
    Compute quantization parameters by taking the moving average of all min/max values

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    def get_current_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_min_max(observed)


def _get_min_max(observed: torch.Tensor) -> MinMaxTuple:
    min_vals = torch.amin(observed, dim=(0, -1))
    max_vals = torch.amax(observed, dim=(0, -1))

    return min_vals, max_vals
