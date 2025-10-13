from abc import abstractmethod
from typing import Optional

import torch
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
        module: Optional[torch.nn.Module] = None,
        **observer_kwargs,
    ):
        super().__init__(base_name, args, module, **observer_kwargs)
        self.avg_constant = self.args.observer_kwargs.get("averaging_constant", 0.01)

        self.past_min_vals = None
        self.past_max_vals = None
        self.past_global_min_vals = None
        self.past_global_max_vals = None

    @abstractmethod
    def get_current_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Calculate the min and max value of the observed value (without moving average)
        """
        raise NotImplementedError()

    @abstractmethod
    def get_current_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Calculate the min and max value of the observed value (without moving average)
        for the purposes of global scale calculation
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

        if self.past_min_vals is not None and self.avg_constant != 1.0:
            # FUTURE: consider scaling by num observations (first dim)
            #         rather than reducing by first dim
            min_vals = self._lerp(self.past_min_vals, min_vals, self.avg_constant)
            max_vals = self._lerp(self.past_max_vals, max_vals, self.avg_constant)

        self.past_min_vals = min_vals
        self.past_max_vals = max_vals

        return min_vals, max_vals

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Calculate moving average of min and max values from observed value
        for the purposes of global scale calculation

        :param observed: value being observed whose shape is
            (num_observations, 1, group_size)
        :return: minimum value and maximum value whose shapes are (1, )
        """
        min_vals, max_vals = self.get_current_global_min_max(observed)

        if self.past_global_min_vals is not None and self.avg_constant != 1.0:
            # FUTURE: consider scaling by num observations (first dim)
            #         rather than reducing by first dim
            min_vals = self._lerp(
                self.past_global_min_vals, min_vals, self.avg_constant
            )
            max_vals = self._lerp(
                self.past_global_max_vals, max_vals, self.avg_constant
            )

        self.past_global_min_vals = min_vals
        self.past_global_max_vals = max_vals

        return min_vals, max_vals

    def _lerp(
        self, input: torch.Tensor, end: torch.Tensor, weight: float
    ) -> torch.Tensor:
        """torch lerp_kernel is not implemeneted for all data types"""
        return (input * (1.0 - weight)) + (end * weight)
