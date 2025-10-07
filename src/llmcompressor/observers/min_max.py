from typing import Optional, Tuple

import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.observers.base import Observer

__all__ = ["MinMaxObserver"]


@Observer.register("minmax")
class MinMaxObserver(Observer):
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

        observer_kwargs = self.args.observer_kwargs
        self.averaging_constant = observer_kwargs.get("averaging_constant", 0.01)

    def get_min_max(self, observed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates updated scales and zero points from observed value using the absolute
        min and max value. If `averaging_constant` is specified, then subsequent calls
        will affect a moving average by the specified constant.

        :param observed: value being observed whose shape is
            (num_observations, *qparam_shape, group_size)
        :return: minimum value and maximum value whose shapes are (*qparam_shape, )
        """
        min_vals = torch.amin(observed, dim=(0, -1))
        max_vals = torch.amax(observed, dim=(0, -1))

        if self.min_vals is not None and self.averaging_constant != 1.0:
            # FUTURE: consider scaling by num observations (first dim)
            #         rather than reducing by first dim
            min_vals = self._lerp(self.min_vals, min_vals, self.averaging_constant)
            max_vals = self._lerp(self.max_vals, max_vals, self.averaging_constant)

        return min_vals, max_vals

    def _lerp(
        self, input: torch.Tensor, end: torch.Tensor, weight: float
    ) -> torch.Tensor:
        """torch lerp_kernel is not implemeneted for all data types"""
        return (input * (1.0 - weight)) + (end * weight)
