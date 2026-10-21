import warnings

import torch
from compressed_tensors.quantization.utils.mxfp_utils import (
    should_generate_mx_scales,
)
from torch import distributed as dist

from llmcompressor.observers.base import Observer
from llmcompressor.observers.grid_search import (
    _default_triton_error_buffer,
    _grid_search_observer,
    _resolve_grid_type,
)
from llmcompressor.observers.helpers import lerp

__all__ = ["MovingAverageMSEObserver"]


@Observer.register("memoryless_mse")
class MemorylessMSEObserver(Observer):
    """
    Search candidate min/max ranges independently for each observed tensor.

    For each candidate range, the observer computes the powered QDQ error
    ``abs(QDQ(w) - w) ** norm`` and sums it over each quantization group. When
    ``magnitude_normalization`` is enabled, each element's error is also
    multiplied by ``sqrt(w**2 + group_absmax**2)``. The selected range is the
    candidate with the lowest error. This observer does not average statistics
    across calls; :class:`MovingAverageMSEObserver` smooths the selected ranges.

    The options below are read from ``QuantizationArgs.observer_kwargs``. The
    expanded MSE observer inherits these options and overrides some defaults.

    Observer kwargs:
        maxshrink (float): Maximum fraction of the expanded range to shrink.
            The linear grid tests multipliers ``1 - i / grid`` for
            ``i < maxshrink * grid``. For the exponential grid, candidates are
            powers of two starting at 1 and stop before shrinking past
            ``1 - maxshrink``. Defaults to ``0.20``.
        patience (int): Number of consecutive candidates outside the current
            best-score tolerance before early stopping. Triton uses
            ``triton_error_buffer`` for this tolerance; eager search counts
            strict non-improvements. Defaults to ``5``.
        grid (float): Search resolution and candidate budget. For a linear
            grid, adjacent multipliers differ by ``1 / grid``. For an
            exponential grid, this is an upper bound on the number of
            candidates. Defaults to ``100``.
        norm (float): Power applied to the absolute QDQ error. ``2`` gives
            squared error; defaults to ``2.4``.
        expand (float): Multiplier applied to the observed min/max range before
            shrink candidates are searched. Must be at least ``1``. Defaults
            to ``1``.
        grid_type (str or None): ``"linear"`` or ``"exponential"``. If
            ``None``, uses exponential spacing when MX scales are generated
            (E8M0) and linear spacing otherwise. Exponential candidates are
            ``1, 1/2, 1/4, ...`` times the expanded range. Defaults to ``None``.
        magnitude_normalization (bool): If true, multiply each powered error
            by ``sqrt(w**2 + group_absmax**2)``, where ``group_absmax`` is the
            largest absolute observed value in that quantization group.
            Defaults to ``False``.
        triton_error_buffer (float): Relative tolerance used by the Triton
            implementation when counting candidates toward per-group
            ``patience``. A candidate within ``1 + triton_error_buffer`` times
            the best error resets patience; it does not change the score or
            selected minimum directly. Defaults to ``1.0`` for 4-bit floating
            point quantization and ``0.30`` otherwise. The eager implementation
            does not use this option.
        averaging_constant (float): Additional option for the moving-average
            ``mse`` observer only. Controls how much each newly selected range
            contributes to the running range; defaults to ``0.01``.
    """

    _act_sync_dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.patience = observer_kwargs.get("patience", 5)
        self.grid = observer_kwargs.get("grid", 100.0)
        self.norm = observer_kwargs.get("norm", 2.4)
        self.magnitude_normalization = observer_kwargs.get(
            "magnitude_normalization", False
        )
        self.grid_type = _resolve_grid_type(observer_kwargs.get("grid_type"), self.args)
        self.triton_error_buffer = observer_kwargs.get(
            "triton_error_buffer", _default_triton_error_buffer(self.args)
        )
        self.expand = observer_kwargs.get("expand", 1.0)
        if self.expand < 1.0:
            raise ValueError(f"expand value must be at least 1.0, got {self.expand}")
        if not isinstance(self.magnitude_normalization, bool):
            raise ValueError(
                "magnitude_normalization must be a bool, got "
                f"{self.magnitude_normalization!r}"
            )

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        self.min_vals, self.max_vals = _grid_search_observer(
            observed,
            self.args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            self.triton_error_buffer,
            expand=self.expand,
            magnitude_normalization=self.magnitude_normalization,
            grid_type=self.grid_type,
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
        self.magnitude_normalization = observer_kwargs.get(
            "magnitude_normalization", False
        )
        self.grid_type = _resolve_grid_type(observer_kwargs.get("grid_type"), self.args)
        self.triton_error_buffer = observer_kwargs.get(
            "triton_error_buffer", _default_triton_error_buffer(self.args)
        )
        self.expand = observer_kwargs.get("expand", 1.0)
        if self.expand < 1.0:
            raise ValueError(f"expand value must be at least 1.0, got {self.expand}")
        if not isinstance(self.magnitude_normalization, bool):
            raise ValueError(
                "magnitude_normalization must be a bool, got "
                f"{self.magnitude_normalization!r}"
            )

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        min_vals, max_vals = _grid_search_observer(
            observed,
            self.args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            self.triton_error_buffer,
            expand=self.expand,
            magnitude_normalization=self.magnitude_normalization,
            grid_type=self.grid_type,
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
        if should_generate_mx_scales(self.args):
            self.expand = observer_kwargs.get("expand", 1.0)
            self.maxshrink = observer_kwargs.get("maxshrink", 1.0 - 1.0 / 64.0)
        else:
            self.expand = observer_kwargs.get("expand", 1.8)
            self.maxshrink = observer_kwargs.get("maxshrink", 1 - 0.8 / 1.8)
        self.grid = observer_kwargs.get("grid", 200.0)
        self.patience = observer_kwargs.get("patience", 1000)
        self.triton_error_buffer = observer_kwargs.get("triton_error_buffer", 1.00)


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
