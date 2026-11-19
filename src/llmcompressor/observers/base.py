from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, TypedDict

import torch
from compressed_tensors import InternalModule
from compressed_tensors.offload.dist_utils import as_broadcastable
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam
from compressed_tensors.registry.registry import RegistryMixin
from torch import distributed as dist

from llmcompressor.observers.fusion import FusionHandler
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

    # Attribute names that constitute this observer's statistics
    _stats_attrs: List[str] = ["min_vals", "max_vals"]

    # If True, activation and attention observations are split along the batch
    # dimension and statistics are updated once per sample, in batch order. This
    # makes order-dependent statistics (e.g. moving averages) independent of the
    # calibration batch size
    _update_per_sample: bool = False

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

        self.fusion_handler = FusionHandler(self)

    @property
    def has_statistics(self) -> bool:
        return all(hasattr(self, attr) for attr in self._stats_attrs)

    @property
    def is_weight_obs(self) -> bool:
        return self.base_name == "weight"

    @abstractmethod
    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        """
        Update internal observer statistics (min_vals, max_vals) from observed tensor.

        :param observed: flattened observed value of shape
                        (num_observations, *qparam_shape, group_size)
        """
        raise NotImplementedError()

    def get_global_scale(self) -> Optional[torch.Tensor]:
        """
        Compute global_scale from this observer and all fused observers.

        Uses the FusionHandler to collect statistics from the fusion group.
        Returns None if strategy is not TENSOR_GROUP.

        :return: global_scale tensor or None
        """
        if self.args.strategy != QuantizationStrategy.TENSOR_GROUP:
            return None

        all_stats = self.fusion_handler.get_fused_statistics()
        global_absmax = all_stats[0]["max_vals"].max()
        for stats in all_stats:
            global_absmax = torch.max(global_absmax, -stats["min_vals"].min())
            global_absmax = torch.max(global_absmax, stats["max_vals"].max())

        return generate_gparam(-global_absmax.reshape(1), global_absmax.reshape(1))

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

        global_scale = self.get_global_scale()

        scale, zero_point = calculate_qparams(
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            quantization_args=self.args,
            global_scale=global_scale,
        )

        self.delete_statistics()

        return {"scale": scale, "zero_point": zero_point, "global_scale": global_scale}

    @torch.no_grad
    def forward(self, observed: torch.Tensor) -> "Observer":
        """
        Update observer statistics from observed value.
        If fused, also triggers observation for all fused observers.

        :param observed: value being observed
        :return: self for method chaining
        """
        if observed.numel() == 0:
            return self

        if self.is_weight_obs and self.has_statistics:
            return self

        num_samples = self._get_num_samples(observed)
        observed = flatten_for_calibration(observed, self.base_name, self.args)
        if num_samples > 1:
            # flattening keeps observations batch-major, so this splits by sample
            for sample in observed.unflatten(0, (num_samples, -1)):
                self.update_statistics_from_observed(sample)
        else:
            self.update_statistics_from_observed(observed)

        self.fusion_handler.get_fused_statistics()

        return self

    def _get_num_samples(self, observed: torch.Tensor) -> int:
        """
        Number of samples to split an observation into before updating statistics.

        Activations of shape (batch_size, seq_len, hidden_dim) and attention states
        of shape (batch_size, num_heads, seq_len, head_dim) are split along the batch
        dimension. Weights, and inputs without a batch dimension (such as the routed
        tokens of an MoE expert), are observed as a whole.

        :param observed: value being observed, before flattening
        :return: number of samples, 1 if the value should not be split
        """
        if not self._update_per_sample or self.is_weight_obs:
            return 1

        batch_ndim = 4 if self.base_name in ("q", "k", "v") else 3
        if observed.ndim < batch_ndim:
            return 1

        return observed.size(0)

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

    def delete_statistics(self, check_fused: bool = True) -> None:
        """
        Delete this observer's statistics.

        :param check_fused: if True (default), use cooperative fusion deletion
            so stats are only removed when all fused observers are ready.
            If False, delete immediately regardless of fusion state.
        """
        if check_fused:
            self.fusion_handler.maybe_delete_statistics()
        else:
            for attr in self._stats_attrs:
                if hasattr(self, attr):
                    delattr(self, attr)

    def get_statistics(self) -> dict[str, torch.Tensor]:
        """
        Return this observer's current statistics.

        :return: dict mapping each stats attr name to its tensor
        """
        assert (
            self.has_statistics
        ), "No statistics available. Call observer(value) first."
        return {attr: getattr(self, attr) for attr in self._stats_attrs}

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
