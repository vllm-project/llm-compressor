from abc import abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple, TypedDict
from weakref import ref

import torch
from compressed_tensors import InternalModule
from compressed_tensors.offload.dist_utils import as_broadcastable
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.registry.registry import RegistryMixin
from torch import distributed as dist
from torch.nn import Module

from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = ["Observer", "MinMaxTuple", "QParamsDict"]


MinMaxTuple = Tuple[torch.Tensor, torch.Tensor]


class QParamsDict(TypedDict, total=False):
    """Dictionary containing quantization parameters."""

    scale: torch.Tensor
    zero_point: torch.Tensor
    global_scale: Optional[torch.Tensor]


_msg = "Fused module has been garbage collected before its weight was observed"


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

        self._fusions: dict["Observer", ref[Module]] = {}

    @property
    def has_statistics(self) -> bool:
        return hasattr(self, "min_vals")

    @abstractmethod
    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        """
        Update internal observer statistics (min_vals, max_vals) from observed tensor.

        :param observed: flattened observed value of shape
                        (num_observations, *qparam_shape, group_size)
        """
        raise NotImplementedError()

    @torch.no_grad
    def get_qparams(self) -> QParamsDict:
        """
        Compute quantization parameters from accumulated statistics.

        For TENSOR_GROUP, global_scale is computed from the absmax of
        this observer and all fused observers.

        :return: dict with keys "scale", "zero_point", and "global_scale"
        """
        assert (
            self.has_statistics
        ), "No statistics available. Call observer(value) first."

        global_scale = None

        if self.args.strategy == QuantizationStrategy.TENSOR_GROUP:
            global_absmax = torch.max(-self.min_vals.min(), self.max_vals.max())
            for fused_obs in self._fusions.keys():
                if not fused_obs.has_statistics:
                    fused_mod = self._fusions[fused_obs]()
                    assert fused_mod is not None, _msg
                    fused_obs(fused_mod.weight)
                global_absmax = torch.max(global_absmax, -fused_obs.min_vals.min())
                global_absmax = torch.max(global_absmax, fused_obs.max_vals.max())
            global_scale = generate_gparam(
                -global_absmax.reshape(1), global_absmax.reshape(1)
            )

        scale, zero_point = calculate_qparams(
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            quantization_args=self.args,
            global_scale=global_scale,
        )

        del self.min_vals
        del self.max_vals

        return {"scale": scale, "zero_point": zero_point, "global_scale": global_scale}

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
        self.update_statistics_from_observed(observed)
        return self

    @staticmethod
    def fuse(observers_and_modules: Iterable[tuple["Observer", Module]]) -> None:
        """
        Link all observers in the list with each other for shared global_scale.
        Capture module weak-references for auto-observation in get_qparams().

        :param observers_and_modules: list of (observer, module) tuples
        """
        pairs = list(observers_and_modules)
        for obs, _ in pairs:
            for fuse_obs, fuse_mod in pairs:
                if fuse_obs is not obs:
                    obs._fusions[fuse_obs] = ref(fuse_mod)

    def sync_activation_stats(self) -> List[dist.Work]:
        """All-reduce accumulated activation statistics across DDP ranks.

            note: weight statistics don't need to be synced since weights
        are synced across ranks, only data (activations) differs by rank.

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
