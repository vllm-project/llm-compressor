import torch
from compressed_tensors.quantization import QuantizationStrategy
from compressed_tensors.quantization.utils import generate_gparam
from torch import distributed as dist

from llmcompressor.observers.base import Observer
from llmcompressor.observers.helpers import lerp
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
        self.use_global_scale_prior = observer_kwargs.get(
            "use_global_scale_prior", False
        )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        # Pre-create token_args to avoid patch_attr context manager
        # which causes torch.compile graph breaks
        self._token_args = self.args.model_copy(
            update={"strategy": QuantizationStrategy.TOKEN}
        )

    def _compute_approx_global_scale(
        self, observed: torch.Tensor
    ) -> torch.Tensor | None:
        if (
            not self.use_global_scale_prior
            or self.args.strategy != QuantizationStrategy.TENSOR_GROUP
        ):
            return None
        absmax = observed.abs().max()
        absmax = torch.clamp(absmax, min=torch.finfo(absmax.dtype).tiny)
        return generate_gparam(-absmax.reshape(1), absmax.reshape(1))

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        gs_prior = self._compute_approx_global_scale(observed)
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
            global_scale=gs_prior,
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


@Observer.register("nvfp4_expanded_mse")
class NVFP4ExpandedMSEObserver(MemorylessMSEObserver):
    """
    MSE observer with defaults tuned for NVFP4 range expansion.

    Searches from ``expand`` times the observed range down to
    ``(1 - maxshrink) * expand`` times the observed range.
    With the defaults below, this covers 1.8x down to ~0.8x of
    the original per-group range in 112 search steps.

    Usage::

        QuantizationArgs(
            ...
            observer="nvfp4_expanded_mse",
        )
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.expand = observer_kwargs.get("expand", 1.8)
        self.maxshrink = observer_kwargs.get("maxshrink", 1 - 0.8 / 1.8)
        self.grid = observer_kwargs.get("grid", 200.0)
        self.patience = observer_kwargs.get("patience", 1000)
