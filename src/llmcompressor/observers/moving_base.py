import torch
from compressed_tensors.quantization import QuantizationArgs
from torch import distributed as dist

from llmcompressor.observers.base import Observer

__all__ = ["MovingAverageObserverBase"]


class MovingAverageObserverBase(Observer):
    """
    Base class for observers that use exponential moving average of statistics.

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    _act_sync_dict = {
        "min_vals": dist.ReduceOp.AVG,
        "max_vals": dist.ReduceOp.AVG,
    }

    def __init__(
        self,
        base_name: str,
        args: QuantizationArgs,
        **observer_kwargs,
    ):
        super().__init__(base_name, args, **observer_kwargs)
        self.avg_constant = self.args.observer_kwargs.get("averaging_constant", 0.01)

    def _lerp(
        self, input: torch.Tensor, end: torch.Tensor, weight: float
    ) -> torch.Tensor:
        """torch lerp_kernel is not implemeneted for all data types"""
        return (input * (1.0 - weight)) + (end * weight)
