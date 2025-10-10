from abc import abstractmethod
from typing import Optional, Tuple
from weakref import ref

import torch
from compressed_tensors import InternalModule
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
)
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.registry.registry import RegistryMixin

from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = ["Observer"]


class Observer(InternalModule, RegistryMixin):
    """
    Base Observer class to be subclassed for specific implementation.
    Subclasses should override `calculate_qparams` to return a scale, zero_point
    pair
    """

    def __init__(
        self,
        base_name: str,
        args: QuantizationArgs,
        module: Optional[torch.nn.Module] = None,
        **observer_kwargs,
    ):
        super().__init__()
        self.module = ref(module) if module is not None else None
        self.base_name = base_name
        self.args = args

        # populate observer kwargs
        self.args.observer_kwargs = self.args.observer_kwargs or {}
        self.args.observer_kwargs.update(observer_kwargs)

        # used for moving averages and testing
        self.min_vals = None
        self.max_vals = None
        self.global_min_vals = None
        self.global_max_vals = None

    @abstractmethod
    def get_min_max(self, observed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate min and max values from observed value

        :param observed: value being observed whose shape is
            (num_observations, *qparam_shape, group_size)
        :return: minimum value and maximum value whose shapes are (*qparam_shape, )
        """
        raise NotImplementedError()

    @abstractmethod
    def get_global_min_max(
        self, observed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate min and max values from observed value for the purposes of
        global scale calculation

        :param observed: value being observed whose shape is
            (num_observations, 1, group_size)
        :return: minimum value and maximum value whose shapes are (1, )
        """
        raise NotImplementedError()

    def forward(self, observed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate updated scales and zero points from observed value
        (weight, activation, or attention state).

        :param observed: value being observed
        :return: calibrated scale and zero point
        """
        g_idx = self._get_module_param("g_idx")
        global_scale = self._get_module_param("global_scale")

        observed = flatten_for_calibration(observed, self.base_name, self.args, g_idx)
        self.min_vals, self.max_vals = self.get_min_max(observed)

        return calculate_qparams(
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            quantization_args=self.args,
            global_scale=global_scale,
        )

    def get_global_scale(self, observed: torch.Tensor) -> torch.nn.Parameter:
        """
        Calculate updated global scale from observed value

        :param observed: value being observed
        :return: calibrated global parameter
        """
        # avoid updating running min/max for global scales
        observed = observed.reshape((1, 1, -1))  # per tensor reshape
        self.global_min_vals, self.global_max_vals = self.get_global_min_max(observed)
        return generate_gparam(self.global_min_vals, self.global_max_vals)

    def _get_module_param(self, name: str) -> Optional[torch.nn.Parameter]:
        if self.module is None:
            return None

        return getattr(self.module(), f"{self.base_name}_{name}", None)
