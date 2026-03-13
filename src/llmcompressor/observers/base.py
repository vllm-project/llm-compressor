from abc import abstractmethod
from typing import Optional, Tuple

import torch
from compressed_tensors import InternalModule
from compressed_tensors.offload import update_offload_parameter
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.registry.registry import RegistryMixin
from torch import distributed as dist

from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = ["Observer", "MinMaxTuple"]

MinMaxTuple = Tuple[torch.Tensor, torch.Tensor]


class Observer(InternalModule, RegistryMixin):
    """
    Base class for observers which compute quantization parameters given observerations
    of weights, activations, or attention states.

    Example:
    ```python
    module = ...
    observer = Observer.load_from_registry(observer, base_name="weight", args=...)
    observer(module.weight)  # accumulate min/max from observation
    qparams = observer.calculate_qparams()  # calculate scales, zero_points, etc.
    observer.calibrate_module(module)  # update module with qparams
    ```

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    min_vals: torch.Tensor | None
    max_vals: torch.Tensor | None

    def __init__(
        self,
        base_name: str,
        args: QuantizationArgs,
        **observer_kwargs,
    ):
        super().__init__()
        self.base_name = base_name
        self.args = args

        self.min_vals = None
        self.max_vals = None

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

    def forward(self, observed: torch.Tensor, g_idx: Optional[torch.Tensor] = None):
        """
        Accumulate min and max values from observed value
        (weight, activation, or attention state).

        :param observed: value being observed
        :param g_idx: optional group index tensor for group quantization
        """
        observed = flatten_for_calibration(observed, self.base_name, self.args, g_idx)
        self.min_vals, self.max_vals = self.get_min_max(observed)

    def calibrate_module(self, module: torch.nn.Module):
        """
        Calculate quantization parameters and update the module with them.

        :param module: module to update with quantization parameters
        """
        for name, value in self.calculate_qparams().items():
            update_offload_parameter(module, name, value)

    def calculate_qparams(
        self, global_scale: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """
        Calculate quantization parameters from accumulated min/max values.

        Calculates and returns global_scale if the quantization strategy requires it
        and no global_scale argument is provided. Otherwise, the global_scale argument
        is used.

        :param global_scale: optional pre-computed global scale for tensor-group quantization
        :return: dictionary mapping parameter names to their computed values
            (e.g., "weight_scale", "weight_zero_point", "weight_global_scale")
        """
        if self.min_vals is None or self.max_vals is None:
            raise ValueError()

        qparams = dict()

        if (
            global_scale is None
            and self.args.strategy == QuantizationStrategy.TENSOR_GROUP
        ):
            global_scale = generate_gparam(self.min_vals.min(), self.max_vals.max())
            qparams[self.base_name + "_global_scale"] = global_scale

        if self.args.dynamic == False:
            scale, zero_point = calculate_qparams(
                min_vals=self.min_vals,
                max_vals=self.max_vals,
                quantization_args=self.args,
                global_scale=global_scale,
            )
            qparams[self.base_name + "_scale"] = scale
            qparams[self.base_name + "_zero_point"] = zero_point

        return qparams

    def synchronize(self) -> list[dist.Work]:
        """
        All-reduce accumulated min/max statistics across DDP ranks

        :return: list of async communication handles
        """
        comms = []
        if self.min_vals is not None:
            comms.append(
                dist.all_reduce(self.min_vals, op=dist.ReduceOp.MIN, async_op=True)
            )
        if self.max_vals is not None:
            comms.append(
                dist.all_reduce(self.max_vals, op=dist.ReduceOp.MAX, async_op=True)
            )

        return comms
