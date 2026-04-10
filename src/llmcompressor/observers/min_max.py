import torch
from typing import Optional

from compressed_tensors.quantization import QuantizationStrategy
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from llmcompressor.observers.base import MinMaxTuple, Observer, QParamsDict
from llmcompressor.observers.moving_base import MovingAverageObserverBase
from torch import distributed as dist

__all__ = ["MemorylessMinMaxObserver", "StaticMinMaxObserver", "MinMaxObserver"]


@Observer.register("memoryless_minmax")
class MemorylessMinMaxObserver(Observer):
    """
    Compute quantization parameters by taking the min/max of the observed value

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    is_memoryless = True
    _sync_dict = {}  # Memoryless - no DDP sync needed

    def _update_statistics(self, observed: torch.Tensor) -> None:
        """Compute and store min/max statistics from observation."""
        # Compute per-group/channel min/max
        self.min_vals, self.max_vals = _get_min_max(observed)


@Observer.register("static_minmax")
class StaticMinMaxObserver(MemorylessMinMaxObserver):
    """
    Compute quantization parameters by taking the min/max of all observed values

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    is_memoryless = False
    _sync_dict = {
        "min_vals": dist.ReduceOp.MIN,
        "max_vals": dist.ReduceOp.MAX,
    }

    def _update_statistics(self, observed: torch.Tensor) -> None:
        """Update accumulated global min/max statistics."""
        # Update per-group/channel min/max
        min_vals, max_vals = _get_min_max(observed)

        if hasattr(self, 'min_vals'):
            self.min_vals = torch.min(min_vals, self.min_vals)
            self.max_vals = torch.max(max_vals, self.max_vals)
        else:
            self.min_vals = min_vals
            self.max_vals = max_vals


@Observer.register("minmax")
class MinMaxObserver(MovingAverageObserverBase):
    """
    Compute quantization parameters by taking the moving average of all min/max values

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    def get_current_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_min_max(observed)


def _get_min_max(observed: torch.Tensor) -> MinMaxTuple:
    min_vals = torch.amin(observed, dim=(0, -1))
    max_vals = torch.amax(observed, dim=(0, -1))

    return min_vals, max_vals
