from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, TypedDict
from weakref import ref
import warnings

import torch
from compressed_tensors import InternalModule
from compressed_tensors.offload.dist_utils import as_broadcastable
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.registry.registry import RegistryMixin
from compressed_tensors.utils import align_module_device
from torch import distributed as dist

from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = ["Observer", "MinMaxTuple", "QParamsDict"]

MinMaxTuple = Tuple[torch.Tensor, torch.Tensor]


class QParamsDict(TypedDict, total=False):
    """Dictionary containing quantization parameters."""
    scale: torch.Tensor
    zero_point: torch.Tensor
    global_scale: Optional[torch.Tensor]


class Observer(InternalModule, RegistryMixin):
    """
    Base class for observers which compute quantization parameters given observerations
    of weights, activations, or attention states.

    Example:
    ```python
    module = ...
    observer = Observer.load_from_registry(observer, base_name="weight", args=...)
    qparams = observer(module.weight).get_qparams()
    scale, zero_point, global_scale = qparams["scale"], qparams["zero_point"], qparams["global_scale"]
    ```

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    # Class attribute indicating whether this observer accumulates statistics across calls
    # Memoryless observers should override this to False
    accumulates_statistics: bool = True

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
        self.statistics = {}  # Dictionary to store observer statistics

        # populate observer kwargs
        self.args.observer_kwargs = self.args.observer_kwargs or {}
        self.args.observer_kwargs.update(observer_kwargs)

    @abstractmethod
    def _update_statistics(self, observed: torch.Tensor) -> None:
        """
        Update internal observer statistics from observed tensor.
        This method should update the observer's internal state (stored in self.statistics)
        with min/max values.

        :param observed: flattened observed value of shape
                        (num_observations, *qparam_shape, group_size)
        """
        raise NotImplementedError()

    @abstractmethod
    def _compute_qparams_from_statistics(self) -> QParamsDict:
        """
        Compute scale and zero_point from accumulated internal statistics.

        :return: dict with keys "scale" and "zero_point"
        """
        raise NotImplementedError()

    @abstractmethod
    def _compute_gparams_from_statistics(self) -> torch.Tensor:
        """
        Compute global_scale from accumulated internal statistics.

        :return: global_scale tensor
        """
        raise NotImplementedError()

    @torch.no_grad
    def get_qparams(self) -> QParamsDict:
        """
        Compute quantization parameters from accumulated statistics.

        For TENSOR_GROUP strategy, automatically computes global_scale first,
        then uses it to compute per-group scale and zero_point.

        For other strategies, global_scale is None.

        :return: dict with keys "scale", "zero_point", and optionally "global_scale"
        """
        if self.args.strategy == QuantizationStrategy.TENSOR_GROUP:
            global_scale = self._compute_gparams_from_statistics()
            # Store on module so _compute_qparams_from_statistics can use it
            module = self.module() if self.module is not None else None
            if module is not None:
                param_name = f"{self.base_name}_global_scale"
                if hasattr(module, param_name):
                    # Parameter already exists, update it
                    from compressed_tensors.utils import update_offload_parameter
                    update_offload_parameter(module, param_name, global_scale)
                else:
                    # Parameter doesn't exist yet, create it
                    setattr(module, param_name, global_scale)
        else:
            global_scale = None

        qparams = self._compute_qparams_from_statistics()
        qparams["global_scale"] = global_scale
        return qparams

    @torch.no_grad
    def forward(self, observed: torch.Tensor) -> "Observer":
        """
        Update observer statistics from observed value.

        To get quantization parameters, call get_qparams() after this method.
        Can be chained: observer(value).get_qparams()

        :param observed: value being observed (weight, activation, or attention state)
        :return: self for method chaining
        """
        g_idx = self._get_module_param("g_idx")
        observed = flatten_for_calibration(observed, self.base_name, self.args, g_idx)
        self._update_statistics(observed)
        return self

    def _get_module_param(self, name: str) -> Optional[torch.nn.Parameter]:
        if self.module is None or (module := self.module()) is None:
            return None

        with align_module_device(module):
            return getattr(module, f"{self.base_name}_{name}", None)

    def synchronize_statistics(self) -> List[dist.Work]:
        """All-reduce accumulated min/max statistics across DDP ranks.

        Issues async all-reduce operations on any accumulated state
        (min_vals, max_vals, global_min_vals, global_max_vals from statistics dict).
        Memoryless observers return an empty list.

        :return: list of async communication handles
        """
        comms = []
        for key, op in [
            ("min_vals", dist.ReduceOp.MIN),
            ("max_vals", dist.ReduceOp.MAX),
            ("global_min_vals", dist.ReduceOp.MIN),
            ("global_max_vals", dist.ReduceOp.MAX),
        ]:
            val = self.statistics.get(key)
            if val is not None:
                comms.append(
                    dist.all_reduce(as_broadcastable(val), op=op, async_op=True)
                )
        return comms


    def attach(self, module: torch.nn.Module) -> None:
        """
        Called when the observer is attached to a module.
        Subclasses can override to register hooks or initialize state.

        :param module: the module this observer is being attached to
        """
        pass

    def detach(self, module: torch.nn.Module) -> None:
        """
        Called before the observer is deleted from a module.
        Subclasses can override to remove hooks and clean up module attributes.

        :param module: the module this observer is being removed from
        """
        pass

    def _check_has_global_scale(self, global_scale: Optional[torch.nn.Parameter]):
        if (
            self.args.strategy == QuantizationStrategy.TENSOR_GROUP
            and global_scale is None
        ):
            raise ValueError(
                "Cannot compute scale and zero points "
                "without first computing global scale"
            )
