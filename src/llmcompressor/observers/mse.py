import torch
from torch import distributed as dist

from llmcompressor.observers.base import Observer
from llmcompressor.observers.grid_search import _grid_search_observer
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
        patience (int): Number of consecutive candidates without strict
            improvement before early stopping in the eager implementation.
            Triton always evaluates the full grid. ``0`` scans the full grid.
            Defaults to ``5``.
        grid (float): Search resolution and candidate budget. For a linear
            grid, adjacent multipliers differ by ``1 / grid``. For an
            exponential grid, this is an upper bound on the number of
            candidates. Defaults to ``100``.
        norm (float): Power applied to the absolute QDQ error. ``2`` gives
            squared error; defaults to ``2.4``.
        expand (float): Multiplier applied to the observed min/max range before
            shrink candidates are searched. Must be at least ``1``. Defaults
            to ``1``.
        grid_type (str): ``"linear"`` or ``"exponential"``. Defaults to
            ``"linear"``; modifier defaults select exponential spacing for MX
            scales. Exponential candidates are ``1, 1/2, 1/4, ...`` times the
            expanded range.
        magnitude_normalization (bool): If true, multiply each powered error
            by ``sqrt(w**2 + group_absmax**2)``, where ``group_absmax`` is the
            largest absolute observed value in that quantization group.
            Defaults to ``False``.
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
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
        self.grid = observer_kwargs.get("grid", 100.0)
        self.norm = observer_kwargs.get("norm", 2.4)
        self.magnitude_normalization = observer_kwargs.get(
            "magnitude_normalization", False
        )
        self.grid_type = observer_kwargs.get("grid_type", "linear")
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
        self.grid_type = observer_kwargs.get("grid_type", "linear")
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
            expand=self.expand,
            magnitude_normalization=self.magnitude_normalization,
            grid_type=self.grid_type,
        )

        if hasattr(self, "min_vals") and self.avg_constant != 1.0:
            min_vals = lerp(self.min_vals, min_vals, self.avg_constant)
            max_vals = lerp(self.max_vals, max_vals, self.avg_constant)

        self.min_vals = min_vals
        self.max_vals = max_vals
