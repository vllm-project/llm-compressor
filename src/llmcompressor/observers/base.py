from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, TypedDict
from weakref import ref
import warnings
from compressed_tensors.utils import update_offload_parameter
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

    # Class attribute indicating whether this observer is memoryless
    # Memoryless observers should override this to True
    is_memoryless: bool = False

    # Dict of statistic attribute names to reduce operations for DDP synchronization
    # Subclasses should override this to specify which attributes to sync
    # e.g., {"min_vals": dist.ReduceOp.MIN, "max_vals": dist.ReduceOp.MAX}
    _sync_dict: Dict[str, dist.ReduceOp] = {}

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

        # global_scale freezing state
        self._frozen_global_scale = False

    @abstractmethod
    def _update_statistics(self, observed: torch.Tensor) -> None:
        """
        Update internal observer statistics from observed tensor.
        This method should update the observer's statistic attributes
        (e.g., self.min_vals, self.max_vals).

        :param observed: flattened observed value of shape
                        (num_observations, *qparam_shape, group_size)
        """
        raise NotImplementedError()

    def _compute_qparams_from_statistics(self) -> QParamsDict:
        """
        Compute all quantization parameters from accumulated internal statistics.

        Default implementation assumes min_vals and max_vals attributes exist.
        Computes scale, zero_point, and global_scale (if TENSOR_GROUP strategy).
        For non-TENSOR_GROUP strategies, global_scale should be None.

        If global_scale is frozen (via freeze_global_scale()), uses the module's
        existing global_scale instead of recomputing it.

        Subclasses can override if they need custom logic.

        :return: dict with keys "scale", "zero_point", and "global_scale"
        """
        if not hasattr(self, 'min_vals') or not hasattr(self, 'max_vals'):
            raise RuntimeError(
                "No statistics available. Call observer(value) first."
            )

        # Compute or reuse global_scale if TENSOR_GROUP strategy
        if self.args.strategy == QuantizationStrategy.TENSOR_GROUP:
            if self._frozen_global_scale:
                # Try to use existing frozen global_scale from module
                global_scale = self._get_module_param("global_scale")
                if global_scale is None:
                    # Frozen but doesn't exist yet - compute it for the first time
                    # Subsequent calls will use this frozen value
                    global_absmax = torch.max(-self.min_vals.min().reshape(1), self.max_vals.max().reshape(1))
                    global_scale = generate_gparam(-global_absmax, global_absmax)
            else:
                # Compute fresh global_scale from statistics
                global_absmax = torch.max(-self.min_vals.min().reshape(1), self.max_vals.max().reshape(1))
                global_scale = generate_gparam(-global_absmax, global_absmax)
        else:
            global_scale = None

        # Compute scale and zero_point using global_scale
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

        Computes scale, zero_point, and global_scale (for TENSOR_GROUP) all at once.
        If global_scale is computed, it's automatically stored on the module.

        :return: dict with keys "scale", "zero_point", and "global_scale"
        """
        qparams = self._compute_qparams_from_statistics()

        # Store global_scale on module if it was computed
        if qparams.get("global_scale") is not None and self.module and self.module():
            module = self.module()
            param_name = f"{self.base_name}_global_scale"
            if hasattr(module, param_name):
                # Parameter already exists, update it
                update_offload_parameter(module, param_name, qparams["global_scale"])
            else:
                # Parameter doesn't exist yet, create it
                setattr(module, param_name, qparams["global_scale"])

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

    def freeze_global_scale(self) -> None:
        """
        Freeze global_scale computation - the observer will use the module's existing
        global_scale value instead of recomputing it from statistics.

        If global_scale doesn't exist yet when frozen, it will be computed once on
        the first call to get_qparams(), then reused on subsequent calls.

        This is useful when you want to ensure consistency of global_scale across
        multiple quantization passes (e.g., after fusing global_scale across layers
        but before running GPTQ).
        """
        self._frozen_global_scale = True

    def unfreeze_global_scale(self) -> None:
        """
        Unfreeze global_scale computation - the observer will recompute global_scale
        from statistics instead of using the module's existing value.
        """
        self._frozen_global_scale = False

    def synchronize_observer(self) -> List[dist.Work]:
        """All-reduce accumulated statistics across DDP ranks.

        Issues async all-reduce operations on statistic attributes specified in
        _sync_dict. Each attribute is reduced using its specified operation.

        :return: list of async communication handles
        """
        if self.is_memoryless:
            return []  # Memoryless observers don't synchronize

        comms = []
        for attr_name, reduce_op in self._sync_dict.items():
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

    def _check_has_global_scale(self, global_scale: Optional[torch.nn.Parameter]):
        if (
            self.args.strategy == QuantizationStrategy.TENSOR_GROUP
            and global_scale is None
        ):
            raise ValueError(
                "Cannot compute scale and zero points "
                "without first computing global scale"
            )
