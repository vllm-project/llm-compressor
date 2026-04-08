import torch
from typing import Optional

from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from llmcompressor.observers.base import MinMaxTuple, Observer, QParamsDict
from llmcompressor.observers.moving_base import MovingAverageObserverBase

__all__ = ["MemorylessMinMaxObserver", "StaticMinMaxObserver", "MinMaxObserver"]


@Observer.register("memoryless_minmax")
class MemorylessMinMaxObserver(Observer):
    """
    Compute quantization parameters by taking the min/max of the observed value

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    accumulates_statistics = False

    def _update_statistics(self, observed: torch.Tensor) -> None:
        """Compute and store min/max statistics from observation."""
        # Compute per-group/channel min/max
        min_vals, max_vals = _get_min_max(observed)
        self.statistics['min_vals'] = min_vals
        self.statistics['max_vals'] = max_vals

        # Compute global (per-tensor) min/max
        global_observed = observed.reshape((1, 1, -1))
        global_min, global_max = _get_min_max(global_observed)
        self.statistics['global_min_vals'] = global_min
        self.statistics['global_max_vals'] = global_max

    def _compute_qparams_from_statistics(self) -> QParamsDict:
        """Compute scale and zero_point from stored statistics."""
        min_vals = self.statistics.get('min_vals')
        max_vals = self.statistics.get('max_vals')

        if min_vals is None or max_vals is None:
            raise RuntimeError(
                "No statistics available. Call observer(value) first."
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
        """Compute global_scale from stored statistics."""
        global_min = self.statistics.get('global_min_vals')
        global_max = self.statistics.get('global_max_vals')

        if global_min is None or global_max is None:
            raise RuntimeError(
                "No global statistics available. Call observer(value) first."
            )

        return generate_gparam(global_min, global_max)


@Observer.register("static_minmax")
class StaticMinMaxObserver(Observer):
    """
    Compute quantization parameters by taking the min/max of all observed values

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    def _update_statistics(self, observed: torch.Tensor) -> None:
        """Update accumulated global min/max statistics."""
        # Update per-group/channel min/max
        min_vals, max_vals = _get_min_max(observed)

        past_min = self.statistics.get('min_vals')
        past_max = self.statistics.get('max_vals')

        if past_min is not None:
            min_vals = torch.min(min_vals, past_min)
            max_vals = torch.max(max_vals, past_max)

        self.statistics['min_vals'] = min_vals
        self.statistics['max_vals'] = max_vals

        # Update global (per-tensor) min/max
        global_observed = observed.reshape((1, 1, -1))
        global_min, global_max = _get_min_max(global_observed)

        past_global_min = self.statistics.get('global_min_vals')
        past_global_max = self.statistics.get('global_max_vals')

        if past_global_min is not None:
            global_min = torch.min(global_min, past_global_min)
            global_max = torch.max(global_max, past_global_max)

        self.statistics['global_min_vals'] = global_min
        self.statistics['global_max_vals'] = global_max

    def _compute_qparams_from_statistics(self) -> QParamsDict:
        """Compute scale and zero_point from accumulated min/max statistics."""
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
        """Compute global_scale from accumulated min/max statistics."""
        global_min = self.statistics.get('global_min_vals')
        global_max = self.statistics.get('global_max_vals')

        if global_min is None or global_max is None:
            raise RuntimeError(
                "No global statistics accumulated. Call observer(value) first."
            )

        return generate_gparam(global_min, global_max)


@Observer.register("minmax")
class MinMaxObserver(MovingAverageObserverBase):
    """
    Compute quantization parameters by taking the moving average of all min/max values

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    def get_current_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_min_max(observed)

    def get_current_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_min_max(observed)


def _get_min_max(observed: torch.Tensor) -> MinMaxTuple:
    min_vals = torch.amin(observed, dim=(0, -1))
    max_vals = torch.amax(observed, dim=(0, -1))

    return min_vals, max_vals
