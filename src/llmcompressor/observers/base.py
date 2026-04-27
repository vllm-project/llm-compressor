from abc import abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict

import torch
from compressed_tensors import InternalModule
from compressed_tensors.offload.dist_utils import as_broadcastable
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.registry.registry import RegistryMixin
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
    Base class for observers which compute quantization parameters given
    observations of weights, activations, or attention states.

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    # Dict of statistic attribute names to reduce operations for DDP synchronization
    _act_sync_dict: Dict[str, dist.ReduceOp] = {}

    def __init__(
        self,
        base_name: str,
        args: QuantizationArgs,
        **observer_kwargs,
    ):
        super().__init__()
        self.base_name = base_name
        self.args = args

        self.args.observer_kwargs = self.args.observer_kwargs or {}
        self.args.observer_kwargs.update(observer_kwargs)

        self._fused_observers: list["Observer"] = []

    @property
    def has_statistics(self) -> bool:
        return hasattr(self, "min_vals")

    @abstractmethod
    def update_statistics(self, observed: torch.Tensor) -> None:
        """
        Update internal observer statistics (min_vals, max_vals) from observed tensor.

        :param observed: flattened observed value of shape
                        (num_observations, *qparam_shape, group_size)
        """
        raise NotImplementedError()

    def compute_qparams_from_statistics(self) -> QParamsDict:
        """
        Compute quantization parameters from accumulated statistics.

        For TENSOR_GROUP, global_scale is computed from the absmax of
        this observer and all fused observers. Fused observers must
        already have statistics — call observe_weight on all modules
        before calling get_qparams on any of them.

        :return: dict with keys "scale", "zero_point", and "global_scale"
        """
        assert (
            self.has_statistics
        ), "No statistics available. Call observer(value) first."

        global_scale = None
        if self.args.strategy == QuantizationStrategy.TENSOR_GROUP:
            global_absmax = torch.max(-self.min_vals.min(), self.max_vals.max())
            for obs in self._fused_observers:
                assert (
                    obs.has_statistics
                ), "All fused observers must be run before get_qparams."
                global_absmax = torch.max(global_absmax, -obs.min_vals.min())
                global_absmax = torch.max(global_absmax, obs.max_vals.max())
            global_scale = generate_gparam(
                -global_absmax.reshape(1), global_absmax.reshape(1)
            )

        scale, zero_point = calculate_qparams(
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            quantization_args=self.args,
            global_scale=global_scale,
        )

        return {"scale": scale, "zero_point": zero_point, "global_scale": global_scale}

    @torch.no_grad
    def get_qparams(self) -> QParamsDict:
        """
        Compute quantization parameters from accumulated statistics.

        :return: dict with keys "scale", "zero_point", and "global_scale"
        """
        return self.compute_qparams_from_statistics()

    @torch.no_grad
    def forward(self, observed: torch.Tensor) -> "Observer":
        """
        Update observer statistics from observed value.

        :param observed: value being observed
        :return: self for method chaining
        """
        if observed.numel() == 0:
            return self

        observed = flatten_for_calibration(observed, self.base_name, self.args)
        self.update_statistics(observed)
        return self

    @staticmethod
    def fuse(observers: Iterable["Observer"]) -> None:
        """
        Link all observers in the list with each other for shared global_scale.

        :param observers: list of observers to fuse together
        """
        observers = list(observers)
        for obs in observers:
            for other in observers:
                if other is not obs:
                    obs._fused_observers.append(other)

    def sync_activation_stats(self) -> List[dist.Work]:
        """All-reduce accumulated activation statistics across DDP ranks.

            note: weight statistics don't need to be synced since weights
        are synced across ranks, only data differs by rank.

        :return: list of async communication handles
        """
        comms = []
        for attr_name, reduce_op in self._act_sync_dict.items():
            val = getattr(self, attr_name, None)
            if val is not None:
                comms.append(
                    dist.all_reduce(as_broadcastable(val), op=reduce_op, async_op=True)
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
