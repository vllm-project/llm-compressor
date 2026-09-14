import warnings

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch import distributed as dist

from llmcompressor.observers.base import Observer
from llmcompressor.observers.helpers import lerp
from llmcompressor.observers.hierarchical_mse import HierarchicalMSEObserver
from llmcompressor.observers.mse_quant import _grid_search_mse

__all__ = ["MovingAverageMSEObserver"]


@Observer.register("memoryless_mse")
class MemorylessMSEObserver(Observer):
    """
    Compute quantization parameters by finding the optimal min/max values which minimize
    the mean of quantization error squared.
    """

    _act_sync_dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.patience = observer_kwargs.get("patience", 5)
        self.grid = observer_kwargs.get("grid", 100.0)
        self.norm = observer_kwargs.get("norm", 2.4)
        self.chunk_size = observer_kwargs.get("chunk_size", 5)
        self.expand = observer_kwargs.get("expand", 1.0)
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.expand < 1.0:
            raise ValueError(f"expand value must be at least 1.0, got {self.expand}")

        # Pre-create token_args to avoid patch_attr context manager
        # which causes torch.compile graph breaks
        self._token_args = self.args.model_copy(
            update={"strategy": QuantizationStrategy.TOKEN}
        )

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        self.min_vals, self.max_vals = _grid_search_mse(
            observed,
            self.args,
            self._token_args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            self.chunk_size,
            self.expand,
        )


@Observer.register("mse")
class MovingAverageMSEObserver(Observer):
    """
    Compute quantization parameters by finding the optimal min/max values which minimize
    the mean of quantization error squared, with moving average smoothing.
    """

    _act_sync_dict = {
        "min_vals": dist.ReduceOp.AVG,
        "max_vals": dist.ReduceOp.AVG,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avg_constant = self.args.observer_kwargs.get("averaging_constant", 0.01)
        observer_kwargs = self.args.observer_kwargs
        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.patience = observer_kwargs.get("patience", 5)
        self.grid = observer_kwargs.get("grid", 100.0)
        self.norm = observer_kwargs.get("norm", 2.4)
        self.chunk_size = observer_kwargs.get("chunk_size", 5)
        self.expand = observer_kwargs.get("expand", 1.0)
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.expand < 1.0:
            raise ValueError(f"expand value must be at least 1.0, got {self.expand}")

        # Pre-create token_args to avoid patch_attr context manager
        # which causes torch.compile graph breaks
        self._token_args = self.args.model_copy(
            update={"strategy": QuantizationStrategy.TOKEN}
        )

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        min_vals, max_vals = _grid_search_mse(
            observed,
            self.args,
            self._token_args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            self.chunk_size,
            self.expand,
        )

        if hasattr(self, "min_vals") and self.avg_constant != 1.0:
            min_vals = lerp(self.min_vals, min_vals, self.avg_constant)
            max_vals = lerp(self.max_vals, max_vals, self.avg_constant)

        self.min_vals = min_vals
        self.max_vals = max_vals


# Alias fouroversix to the NVFP4ExpandedMSEObserver. Our results show
# this is a more effective way to take advantage of the same range expansion
# benefit that fouroversix is based on.
# Results: https://github.com/vllm-project/llm-compressor/pull/2950
@Observer.register("nvfp4_expanded_mse")
class NVFP4ExpandedMSEObserver(HierarchicalMSEObserver):
    """
    MSE observer with defaults tuned for NVFP4 range expansion.

    Uses a 32-to-32 hierarchical search over 0.8x--1.8x of each original
    per-group range. It retains four coarse candidates per group and refines
    each with eight children, for 64 candidate evaluations per group.

    Usage::

        QuantizationArgs(
            ...
            observer="nvfp4_expanded_mse",
        )
    """


@Observer.register("fouroversix")
def _load_fouroversix_alias(*args, **kwargs) -> NVFP4ExpandedMSEObserver:
    warnings.warn(
        "The 'fouroversix' observer is an alias for 'nvfp4_expanded_mse', "
        "which our results showed to be more accurate at taking advantage "
        "of the same range expansion benefit. See "
        "https://github.com/vllm-project/llm-compressor/pull/2950.",
        UserWarning,
        stacklevel=2,
    )
    return NVFP4ExpandedMSEObserver(*args, **kwargs)
