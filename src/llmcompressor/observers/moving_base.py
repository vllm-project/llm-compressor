from abc import abstractmethod
from typing import List, Optional

import torch
from compressed_tensors.offload.dist_utils import as_broadcastable
from compressed_tensors.quantization.quant_args import QuantizationArgs
from torch import distributed as dist

from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from llmcompressor.observers.base import MinMaxTuple, Observer, QParamsDict

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

    def _update_statistics(self, observed: torch.Tensor) -> None:
        """Update exponential moving average statistics."""
        # Update per-group/channel min/max with EMA
        min_vals, max_vals = self.get_current_min_max(observed)

        past_min = self.statistics.get('min_vals')
        past_max = self.statistics.get('max_vals')

        if past_min is not None and self.avg_constant != 1.0:
            min_vals = self._lerp(past_min, min_vals, self.avg_constant)
            max_vals = self._lerp(past_max, max_vals, self.avg_constant)

        self.statistics['min_vals'] = min_vals
        self.statistics['max_vals'] = max_vals

        # Update global (per-tensor) min/max with EMA
        global_observed = observed.reshape((1, 1, -1))
        global_min, global_max = self.get_current_global_min_max(global_observed)

        past_global_min = self.statistics.get('global_min_vals')
        past_global_max = self.statistics.get('global_max_vals')

        if past_global_min is not None and self.avg_constant != 1.0:
            global_min = self._lerp(past_global_min, global_min, self.avg_constant)
            global_max = self._lerp(past_global_max, global_max, self.avg_constant)

        self.statistics['global_min_vals'] = global_min
        self.statistics['global_max_vals'] = global_max

    def _compute_qparams_from_statistics(self) -> QParamsDict:
        """Compute scale and zero_point from accumulated EMA statistics."""
        min_vals = self.statistics.get('min_vals')
        max_vals = self.statistics.get('max_vals')

        if min_vals is None or max_vals is None:
            raise RuntimeError(
                "No statistics accumulated. Call observer(value) first."
            )

        # Get global_scale from module (set by get_qparams if TENSOR_GROUP)
        global_scale = self._get_module_param("global_scale")
        self._check_has_global_scale(global_scale)

        scale, zero_point = calculate_qparams(
            min_vals=min_vals,
            max_vals=max_vals,
            quantization_args=self.args,
            global_scale=global_scale,
        )

        return {"scale": scale, "zero_point": zero_point}

    def _compute_gparams_from_statistics(self) -> torch.Tensor:
        """Compute global_scale from accumulated EMA statistics."""
        global_min = self.statistics.get('global_min_vals')
        global_max = self.statistics.get('global_max_vals')

        if global_min is None or global_max is None:
            raise RuntimeError(
                "No global statistics accumulated. Call observer(value) first."
            )

        return generate_gparam(global_min, global_max)

    def synchronize_statistics(self) -> List[dist.Work]:
        """Average accumulated moving-average min/max statistics across DDP ranks.

        Unlike :class:`StaticMinMaxObserver` which reduces via MIN/MAX,
        moving-average observers divide by world_size first and then SUM
        so that the result is the average across ranks.

        :return: list of async communication handles
        """
        comms = []
        world_size = dist.get_world_size()
        for key in ("min_vals", "max_vals", "global_min_vals", "global_max_vals"):
            val = self.statistics.get(key)
            if val is not None:
                val.div_(world_size)
                comms.append(
                    dist.all_reduce(
                        as_broadcastable(val), op=dist.ReduceOp.AVG, async_op=True
                    )
                )
        return comms

    def _lerp(
        self, input: torch.Tensor, end: torch.Tensor, weight: float
    ) -> torch.Tensor:
        """torch lerp_kernel is not implemeneted for all data types"""
        return (input * (1.0 - weight)) + (end * weight)
