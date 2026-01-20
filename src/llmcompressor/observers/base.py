from abc import abstractmethod
from typing import Optional, Tuple
from weakref import ref

import torch
from compressed_tensors import InternalModule
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.registry.registry import RegistryMixin
from compressed_tensors.utils import align_module_device

from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = ["Observer", "MinMaxTuple", "ScaleZpTuple"]

MinMaxTuple = Tuple[torch.Tensor, torch.Tensor]
ScaleZpTuple = Tuple[torch.Tensor, torch.Tensor]


class Observer(InternalModule, RegistryMixin):
    """
    Base class for observers which compute quantization parameters given observerations
    of weights, activations, or attention states.

    Example:
    ```python
    module = ...
    observer = Observer.load_from_registry(observer, base_name="weight", args=...)
    module.global_scale = observer.get_global_scale(module.weight)
    scales, zero_points = observer(module.weight)
    ```

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
        super().__init__()
        self.module = ref(module) if module is not None else None
        self.base_name = base_name
        self.args = args

        # populate observer kwargs
        self.args.observer_kwargs = self.args.observer_kwargs or {}
        self.args.observer_kwargs.update(observer_kwargs)

    @abstractmethod
    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Calculate min and max values from observed value

        :param observed: value of shape (num_observations, *qparam_shape, group_size)
        :return: minimum value and maximum value whose shapes are (*qparam_shape, )
        """
        raise NotImplementedError()

    @abstractmethod
    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Calculate min and max values from observed value for the purposes of
        global scale calculation

        :param observed: value of shape (num_observations, 1, group_size)
        :return: minimum value and maximum value whose shapes are (1, )
        """
        raise NotImplementedError()

    @torch.no_grad
    def forward(self, observed: torch.Tensor) -> ScaleZpTuple:
        """
        Calculate updated scales and zero points from observed value
        (weight, activation, or attention state).

        :param observed: value being observed
        :return: calibrated scale and zero point
        """
        scales, zero_points, _min, _max = self._forward_with_minmax(observed)
        return (scales, zero_points)

    @torch.no_grad
    def get_global_scale(self, observed: torch.Tensor) -> torch.Tensor:
        """
        Calculate updated global scale from observed value
        (weight, activation, or attention state).

        :param observed: value being observed
        :return: calibrated global parameter
        """
        global_scale, _min, _max = self._get_global_scale_with_minmax(observed)
        return global_scale

    def _forward_with_minmax(
        self, observed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g_idx = self._get_module_param("g_idx")
        global_scale = self._get_module_param("global_scale")
        self._check_has_global_scale(global_scale)

        observed = flatten_for_calibration(observed, self.base_name, self.args, g_idx)
        min_vals, max_vals = self.get_min_max(observed)

        scales, zero_points = calculate_qparams(
            min_vals=min_vals,
            max_vals=max_vals,
            quantization_args=self.args,
            global_scale=global_scale,
        )
        return scales, zero_points, min_vals, max_vals

    def _get_global_scale_with_minmax(
        self, observed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observed = observed.reshape((1, 1, -1))  # per tensor reshape

        global_min_vals, global_max_vals = self.get_global_min_max(observed)
        global_scale = generate_gparam(global_min_vals, global_max_vals)

        return global_scale, global_min_vals, global_max_vals

    def _get_module_param(self, name: str) -> Optional[torch.nn.Parameter]:
        if self.module is None or (module := self.module()) is None:
            return None

        with align_module_device(module):
            return getattr(module, f"{self.base_name}_{name}", None)

    def _check_has_global_scale(self, global_scale: Optional[torch.nn.Parameter]):
        if (
            self.args.strategy == QuantizationStrategy.TENSOR_GROUP
            and global_scale is None
        ):
            raise ValueError(
                "Cannot compute scale and zero points "
                "without first computing global scale"
            )
