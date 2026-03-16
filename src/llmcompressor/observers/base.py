from abc import abstractmethod
from typing import Optional, Tuple
from weakref import ref

import torch
from compressed_tensors import InternalModule
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.registry.registry import RegistryMixin
from compressed_tensors.utils import align_module_device, update_offload_parameter
from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = ["Observer", "MinMaxTuple", "ScaleZpTuple", "update_module_qparams_from_observer"]

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

    def update_deferred_stats(self, observed: torch.Tensor):
        """
        Accumulate global min/max from an observed tensor into ``_deferred_min``
        and ``_deferred_max`` on this observer.

        Called by ``calibrate_activations`` in ``stats_only`` mode for ALL observer
        types including ``MemorylessMinMaxObserver`` which has no ``past_min_vals``.

        :param observed: activation tensor for this batch
        """
        batch_min = observed.float().min()
        batch_max = observed.float().max()

        if not hasattr(self, "_deferred_min") or self._deferred_min is None:
            self._deferred_min = batch_min
            self._deferred_max = batch_max
        else:
            self._deferred_min = torch.min(self._deferred_min, batch_min)
            self._deferred_max = torch.max(self._deferred_max, batch_max)

    def get_accumulated_min_max(self) -> Optional[MinMaxTuple]:
        """
        Return accumulated min/max populated by ``update_deferred_stats``.
        Returns None if no batches have been seen yet.

        Works for all observer types including ``MemorylessMinMaxObserver``.

        :return: (min_vals, max_vals) tensors or None
        """
        min_vals = getattr(self, "_deferred_min", None)
        max_vals = getattr(self, "_deferred_max", None)
        if min_vals is None or max_vals is None:
            return None
        return min_vals, max_vals

    def clear_accumulated_stats(self):
        """
        Delete accumulated running statistics to free memory after qparams have been
        computed and written to the parent module.
        """
        for attr in (
            "_deferred_min",
            "_deferred_max",
            "past_min_vals",
            "past_max_vals",
            "past_global_min_vals",
            "past_global_max_vals",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    @torch.no_grad
    def forward(self, observed: torch.Tensor) -> ScaleZpTuple:
        """
        Accumulate running statistics from the observed value and update
        deferred min/max. Qparams (scale/zero_point) are not computed here;
        they are written once at epoch end via update_module_qparams_from_observer.

        :param observed: value being observed
        :return: calibrated scale and zero point (from accumulated stats)
        """
        self.update_deferred_stats(observed)
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


@torch.no_grad()
def update_module_qparams_from_observer(
    module: torch.nn.Module,
    base_name: str,
) -> bool:
    """
    Flush an observer's accumulated running statistics into the parent module's
    quantization parameters (scale / zero_point), then free the running stats.

    This is the deferred counterpart to ``call_observer``. Instead of accepting a
    fresh activation tensor, it reads the min/max values that the observer has
    already accumulated across all calibration batches and computes qparams from
    those final statistics.

    :param module: module whose ``{base_name}_observer`` attribute holds the observer
    :param base_name: one of "input", "output", "q", "k", "v"
    :return: True if qparams were updated, False if observer had no accumulated stats
    """
    observer: Optional[Observer] = getattr(module, f"{base_name}_observer", None)
    if observer is None:
        return False

    accumulated = observer.get_accumulated_min_max()
    if accumulated is None:
        return False

    min_vals, max_vals = accumulated
    global_scale = getattr(module, f"{base_name}_global_scale", None)

    with align_module_device(module):
        scales, zero_points = calculate_qparams(
            min_vals=min_vals,
            max_vals=max_vals,
            quantization_args=observer.args,
            global_scale=global_scale,
        )
        update_offload_parameter(module, f"{base_name}_scale", scales)
        if hasattr(module, f"{base_name}_zero_point"):
            update_offload_parameter(module, f"{base_name}_zero_point", zero_points)

    # Free memory — running stats no longer needed
    observer.clear_accumulated_stats()
    return True
