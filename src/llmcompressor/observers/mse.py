from typing import Optional, Tuple

import torch
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.utils import patch_attr

from llmcompressor.observers.base import Observer

__all__ = ["MovingAverageMSEObserver"]


@Observer.register("mse")
class MovingAverageMSEObserver(Observer):
    """
    Implements a dynamic quantization observer that sets the scale and
    zero point based on a moving average of the mse-clipped min and max observed values
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
        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.patience = observer_kwargs.get("patience", 5)
        self.averaging_constant = observer_kwargs.get("averaging_constant", 0.01)
        self.grid = observer_kwargs.get("grid", 100.0)
        self.norm = observer_kwargs.get("norm", 2.4)

    def get_min_max(self, observed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates updated scales and zero points from observed value. Minimum and
        maximum values are chosen by grid searching across min/max values which minimize
        quantization reconstruction loss.

        :param observed: value being observed whose shape is
            (num_observations, *qparam_shape, group_size)
        :return: minimum value and maximum value whose shapes are (*qparam_shape, )
        """
        min_vals, max_vals = self._mse_min_max(observed)

        if self.min_vals is not None and self.averaging_constant != 1.0:
            # FUTURE: consider scaling by num observations (first dim)
            #         rather than reducing by first dim
            min_vals = self._lerp(self.min_vals, min_vals, self.averaging_constant)
            max_vals = self._lerp(self.max_vals, max_vals, self.averaging_constant)

        return min_vals, max_vals

    def _mse_min_max(self, observed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Grid search for MSE-optimal min and max values

        :param observed: value being observed whose shape is
            (num_observations, *qparam_shape, group_size)
        :return: minimum and maximum values which minimize reconstruction error
        """
        from compressed_tensors.quantization.lifecycle import fake_quantize

        absolute_min_val = torch.amin(observed, dim=(0, -1))
        absolute_max_val = torch.amax(observed, dim=(0, -1))
        best = torch.full_like(
            absolute_min_val, torch.finfo(absolute_min_val.dtype).max
        )
        min_val = torch.ones_like(absolute_min_val)
        max_val = torch.zeros_like(absolute_max_val)
        global_scale = self._get_module_param("global_scale")

        # Early stopping params
        no_improve_count = 0

        for i in range(int(self.maxshrink * self.grid)):
            p = 1 - i / self.grid
            shrinked_min_val = p * absolute_min_val
            shrinked_max_val = p * absolute_max_val

            candidate_scales, candidate_zero_points = calculate_qparams(
                min_vals=shrinked_min_val,
                max_vals=shrinked_max_val,
                quantization_args=self.args,
                global_scale=global_scale,
            )

            # Note that observed.shape = (num_observations, *qparams_shape, group_size).
            # For the purposes of fake quantization, this is equivalent to token quant
            with patch_attr(self.args, "strategy", QuantizationStrategy.TOKEN):
                q = fake_quantize(
                    observed,
                    candidate_scales.unsqueeze(-1),
                    candidate_zero_points.unsqueeze(-1),
                    self.args,
                    global_scale=global_scale,
                )

            q -= observed
            q.abs_()
            q.pow_(self.norm)
            err = torch.sum(q, dim=(0, -1))

            tmp = err < best
            if torch.any(tmp):
                best[tmp] = err[tmp]
                min_val[tmp] = shrinked_min_val[tmp]
                max_val[tmp] = shrinked_max_val[tmp]
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    break

        return min_val, max_val

    def _lerp(
        self, input: torch.Tensor, end: torch.Tensor, weight: float
    ) -> torch.Tensor:
        """torch lerp_kernel is not implemeneted for all data types"""
        return (input * (1.0 - weight)) + (end * weight)
