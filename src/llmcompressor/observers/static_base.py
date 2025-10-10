from abc import abstractmethod
from typing import Tuple

import torch

from llmcompressor.observers.base import Observer

__all__ = ["StaticObserverBase"]


class StaticObserverBase(Observer):
    """
    Implements a quantization observer that calculates scale and zero point based on the
    minimum and maximum values of all observed values
    """

    @abstractmethod
    def get_current_min_max(
        self, observed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the min and max value of the observed value
        """
        raise NotImplementedError()

    def get_min_max(self, observed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate min and max values from all observed values

        :param observed: value being observed whose shape is
            (num_observations, *qparam_shape, group_size)
        :return: minimum value and maximum value whose shapes are (*qparam_shape, )
        """
        min_vals, max_vals = self.get_current_min_max(observed)

        if self.min_vals is not None:
            min_vals = torch.min(min_vals, self.min_vals)
            max_vals = torch.max(max_vals, self.max_vals)

        return min_vals, max_vals

    def get_global_min_max(
        self, observed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate min and max values from all observed values for the purposes of global
        scale calculation

        :param observed: value being observed whose shape is
            (num_observations, 1, group_size)
        :return: minimum value and maximum value whose shapes are (1, )
        """
        min_vals, max_vals = self.get_current_min_max(observed)

        if self.global_min_vals is not None:
            min_vals = torch.min(min_vals, self.global_min_vals)
            max_vals = torch.max(max_vals, self.global_max_vals)

        return min_vals, max_vals
