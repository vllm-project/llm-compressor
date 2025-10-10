from abc import abstractmethod
from typing import Optional, Tuple

import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.observers.base import Observer

__all__ = ["MovingAverageObserverBase"]


class MovingAverageObserverBase(Observer):
    """
    Implements a quantization observer that calculates scale and zero point based on the
    minimum and maximum values of the tensor being observed. If averaging_constant is
    specified, then the scales are updated using a moving average
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

    @abstractmethod
    def get_current_min_max(
        self, observed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the min and max value of the observed value (without moving average)
        """
        raise NotImplementedError()

    def get_min_max(self, observed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        return min_vals, max_vals

    def get_global_min_max(
        self, observed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate moving average of min and max values from observed value for the
        purposes of global scale calculation

        :param observed: value being observed whose shape is
            (num_observations, 1, group_size)
        :return: minimum value and maximum value whose shapes are (1, )
        """
        min_vals, max_vals = self.get_current_min_max(observed)

        if self.global_min_vals is not None and self.avg_constant != 1.0:
            # FUTURE: consider scaling by num observations (first dim)
            #         rather than reducing by first dim
            min_vals = self._lerp(self.global_min_vals, min_vals, self.avg_constant)
            max_vals = self._lerp(self.global_max_vals, max_vals, self.avg_constant)

        return min_vals, max_vals

    def _lerp(
        self, input: torch.Tensor, end: torch.Tensor, weight: float
    ) -> torch.Tensor:
        """torch lerp_kernel is not implemeneted for all data types"""
        return (input * (1.0 - weight)) + (end * weight)
