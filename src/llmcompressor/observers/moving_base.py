from abc import abstractmethod

import torch
import torch.distributed as dist
from compressed_tensors.quantization.quant_args import QuantizationArgs

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

    def __init__(
        self,
        base_name: str,
        args: QuantizationArgs,
        **observer_kwargs,
    ):
        super().__init__(base_name, args, **observer_kwargs)
        self.avg_constant = self.args.observer_kwargs.get("averaging_constant", 0.01)

    @abstractmethod
    def get_current_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Calculate the min and max value of the observed value (without moving average)
        """
        raise NotImplementedError()

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Calculate moving average of min and max values from observed value

        :param observed: value being observed whose shape is
            (num_observations, *qparam_shape, group_size)
        :return: minimum value and maximum value whose shapes are (*qparam_shape, )
        """
        min_vals, max_vals = self.get_current_min_max(observed)

        if self.min_vals is not None and self.avg_constant != 1.0:
            # FUTURE: consider scaling by num observations (first dim)
            #         rather than reducing by first dim
            min_vals = self._lerp(self.min_vals, min_vals, self.avg_constant)
            max_vals = self._lerp(self.max_vals, max_vals, self.avg_constant)

        self.min_vals = min_vals
        self.max_vals = max_vals

        return min_vals, max_vals

    def synchronize(self) -> list[dist.Work]:
        comms = []
        if self.min_vals is not None:
            comms.append(
                dist.all_reduce(self.min_vals, op=dist.ReduceOp.AVG, async_op=True)
            )
        if self.max_vals is not None:
            comms.append(
                dist.all_reduce(self.max_vals, op=dist.ReduceOp.AVG, async_op=True)
            )

        return comms

    def _lerp(
        self, input: torch.Tensor, end: torch.Tensor, weight: float
    ) -> torch.Tensor:
        """torch lerp_kernel is not implemeneted for all data types"""
        return (input * (1.0 - weight)) + (end * weight)
