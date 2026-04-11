from abc import abstractmethod
from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationArgs
from torch import distributed as dist

from llmcompressor.observers.base import MinMaxTuple, Observer

__all__ = ["MovingAverageObserverBase"]


class MovingAverageObserverBase(Observer):
    """
    Compute quantization parameters by taking the moving average of min/max values

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    _sync_dict = {
        "min_vals": dist.ReduceOp.AVG,
        "max_vals": dist.ReduceOp.AVG,
    }

    def __init__(
        self,
        base_name: str,
        args: QuantizationArgs,
        module: Optional[torch.nn.Module] = None,
        **observer_kwargs,
    ):
        super().__init__(base_name, args, module, **observer_kwargs)
        self.avg_constant = self.args.observer_kwargs.get("averaging_constant", 0.01)

    @abstractmethod
    def get_current_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Calculate the min and max value of the observed value (without moving average)
        """
        raise NotImplementedError()

    def _update_statistics(self, observed: torch.Tensor) -> None:
        """Update exponential moving average statistics."""
        # Update per-group/channel min/max with EMA
        min_vals, max_vals = self.get_current_min_max(observed)

        if hasattr(self, "min_vals") and self.avg_constant != 1.0:
            min_vals = self._lerp(self.min_vals, min_vals, self.avg_constant)
            max_vals = self._lerp(self.max_vals, max_vals, self.avg_constant)

        self.min_vals = min_vals
        self.max_vals = max_vals

    def _lerp(
        self, input: torch.Tensor, end: torch.Tensor, weight: float
    ) -> torch.Tensor:
        """torch lerp_kernel is not implemeneted for all data types"""
        return (input * (1.0 - weight)) + (end * weight)
