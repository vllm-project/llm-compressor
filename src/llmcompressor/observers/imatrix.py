import math

import torch
from compressed_tensors.quantization import QuantizationStrategy
from compressed_tensors.quantization.utils.mxfp_utils import (
    should_generate_mx_scales,
)
from loguru import logger
from torch import distributed as dist
from torch.utils.hooks import RemovableHandle

from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.observers.base import Observer
from llmcompressor.observers.grid_search import (
    _default_triton_error_buffer,
    _grid_search_observer,
    _resolve_grid_type,
)
from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = ["IMatrixMSEObserver"]

_GROUP_STRATEGIES = (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP)

IMATRIX_PRECISION = torch.float32


@Observer.register("imatrix_mse")
class IMatrixMSEObserver(Observer):
    """
    Search candidate min/max ranges using per-input-channel activation
    importance (E[x²]) to weight QDQ error.

    Supports CHANNEL, GROUP, and TENSOR_GROUP for weight-only Linear modules.
    Falls back to uniform MSE when importance data is unavailable.

    Importance is accumulated on the observer as raw ``_imatrix_sum`` /
    ``_imatrix_count`` and synced across DDP ranks via ``_act_sync_dict``
    before observation.

    The base objective is ``importance * abs(QDQ(w) - w) ** norm``, summed over
    each quantization group. If ``magnitude_normalization`` is enabled, it also
    multiplies each element by ``sqrt(w**2 + group_absmax**2)``. Importance is
    accumulated from observer inputs as E[x²].

    Observer kwargs are read from ``QuantizationArgs.observer_kwargs``:
        maxshrink (float): Maximum fraction of range to shrink. Defaults to
            ``0.95``.
        patience (int): Number of consecutive candidates outside the current
            best-score tolerance before early stopping. Triton uses
            ``triton_error_buffer`` for this tolerance; eager search counts
            strict non-improvements. ``0`` runs the full search. Defaults to
            ``5``.
        grid (float): Linear grid resolution or exponential candidate budget.
            Defaults to ``20``.
        norm (float): Power applied to absolute QDQ error before importance
            weighting. Defaults to ``3.0``.
        expand (float): Multiplier applied to observed min/max before the
            shrink search; must be at least ``1``. Defaults to ``1.0``.
        strict (bool): Raise an error if importance is unavailable or has an
            unsupported shape. If false, warn and fall back to uniform MSE.
            Defaults to ``False``.
        grid_type (str or None): ``"linear"`` or ``"exponential"``. ``None``
            selects exponential spacing when MX scales are generated (E8M0),
            and linear spacing otherwise. Exponential candidates are powers
            of two starting at 1. Defaults to ``None``.
        magnitude_normalization (bool): Multiply each error by
            ``sqrt(w**2 + group_absmax**2)``. Defaults to ``False``.
        triton_error_buffer (float): Relative tolerance used by the Triton
            implementation for per-group patience; it affects early stopping,
            not the error score. Defaults to ``1.0`` for 4-bit floating point
            quantization and ``0.30`` otherwise. The eager implementation does
            not use this option.
    """

    _act_sync_dict = {
        "_imatrix_sum": dist.ReduceOp.SUM,
        "_imatrix_count": dist.ReduceOp.SUM,
    }

    _stats_attrs = ["min_vals", "max_vals", "_imatrix_sum", "_imatrix_count"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kw = self.args.observer_kwargs
        self.maxshrink = kw.get("maxshrink", 0.95)
        self.patience = kw.get("patience", 5)
        self.grid = kw.get("grid", 20)
        self.norm = kw.get("norm", 3.0)
        self.strict = kw.get("strict", False)
        self.expand = kw.get("expand", 1.0)
        self.magnitude_normalization = kw.get("magnitude_normalization", False)
        self.grid_type = _resolve_grid_type(kw.get("grid_type"), self.args)
        self.triton_error_buffer = kw.get(
            "triton_error_buffer", _default_triton_error_buffer(self.args)
        )

        self._imatrix_sum: torch.Tensor | None = None
        self._imatrix_count: torch.Tensor = torch.tensor(0, dtype=torch.int64)
        self._imatrix_hook: RemovableHandle | None = None

        if self.grid <= 0:
            raise ValueError(f"grid must be > 0, got {self.grid}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
        if not (0 <= self.maxshrink <= 1):
            raise ValueError(f"maxshrink must be in [0, 1], got {self.maxshrink}")
        if not isinstance(self.magnitude_normalization, bool):
            raise ValueError(
                "magnitude_normalization must be a bool, got "
                f"{self.magnitude_normalization!r}"
            )
        if (
            not isinstance(self.norm, (int, float))
            or not math.isfinite(self.norm)
            or self.norm <= 0
        ):
            raise ValueError(f"norm must be a finite positive number, got {self.norm}")

    # ------------------------------------------------------------------
    # Hook lifecycle: collect E[x²] per input channel
    # ------------------------------------------------------------------

    def attach(self, module: torch.nn.Module) -> None:
        """Attach a forward-pre hook to accumulate E[x²] per input channel."""
        if self._imatrix_hook is not None:
            self._imatrix_hook.remove()
            self._imatrix_hook = None

        if not hasattr(module, "in_features"):
            return

        in_features = module.in_features
        param = next(module.parameters(), None)
        device = param.device if param is not None else None
        self._imatrix_sum = torch.zeros(
            in_features, dtype=IMATRIX_PRECISION, device=device
        )
        self._imatrix_count = torch.tensor(0, dtype=torch.int64, device=device)

        def _hook(mod, args):
            if (
                HooksMixin._HOOKS_DISABLED
                and getattr(self, "_imatrix_hook", None)
                not in HooksMixin._HOOKS_KEEP_ENABLED
            ):
                return
            x = args[0] if isinstance(args, tuple) else args
            if isinstance(x, tuple):
                x = x[0]
            if x is None or not isinstance(x, torch.Tensor):
                return

            x_f = x.detach().to(IMATRIX_PRECISION)
            device = x_f.device
            n_tokens = math.prod(x_f.shape[:-1])
            token_sum = x_f.pow(2).sum(dim=list(range(x_f.dim() - 1)))

            self._imatrix_sum = self._imatrix_sum.to(device)
            self._imatrix_count = self._imatrix_count.to(device)

            self._imatrix_sum.add_(token_sum)
            self._imatrix_count += n_tokens

        self._imatrix_hook = module.register_forward_pre_hook(_hook)

    def detach(self, module: torch.nn.Module) -> None:
        """Remove the activation collection hook."""
        if self._imatrix_hook is not None:
            self._imatrix_hook.remove()
            self._imatrix_hook = None

    # ------------------------------------------------------------------

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        importance_weights = self._prepare_importance(observed)
        self.min_vals, self.max_vals = _grid_search_observer(
            observed,
            self.args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            self.triton_error_buffer,
            expand=self.expand,
            importance_weights=importance_weights,
            use_imatrix_error=True,
            magnitude_normalization=self.magnitude_normalization,
            grid_type=self.grid_type,
        )

    # ------------------------------------------------------------------

    def _prepare_importance(self, observed: torch.Tensor) -> torch.Tensor | None:
        """Validate → normalize → broadcast to match observed shape."""
        imp = self._get_validated_importance(observed)
        if imp is None:
            return None

        imp = imp.to(device=observed.device, dtype=torch.float32)
        imp = imp / (imp.mean() + torch.finfo(torch.float32).tiny)

        out_features = observed.shape[1]
        imp_2d = imp.unsqueeze(0).expand(out_features, -1)
        return flatten_for_calibration(imp_2d, self.base_name, self.args)

    def _get_validated_importance(self, observed: torch.Tensor) -> torch.Tensor | None:
        """Compute importance from sum/count, validate, and return 1D tensor or None."""
        if self.base_name != "weight":
            if self.strict:
                raise NotImplementedError(
                    "imatrix_mse: only supported for weight observers"
                )
            logger.warning(
                "imatrix_mse: only supported for weight observers."
                " Falling back to uniform MSE.",
                log_once=True,
            )
            return None

        if self.args.strategy == QuantizationStrategy.TENSOR:
            if self.strict:
                raise NotImplementedError("imatrix_mse: TENSOR strategy not supported")
            logger.warning(
                "imatrix_mse: TENSOR strategy not supported."
                " Falling back to uniform MSE.",
                log_once=True,
            )
            return None

        if self._imatrix_sum is None or self._imatrix_count.item() == 0:
            if self.strict:
                raise ValueError("imatrix_mse: no importance data available")
            logger.warning(
                "imatrix_mse: no importance data available."
                " Falling back to uniform MSE.",
                log_once=True,
            )
            return None

        imp = self._imatrix_sum / self._imatrix_count.float()

        if not torch.isfinite(imp).all():
            if self.strict:
                raise ValueError("imatrix_mse: contains non-finite values")
            logger.warning(
                "imatrix_mse: contains non-finite values. Falling back to uniform MSE.",
                log_once=True,
            )
            return None
        if (imp < 0).any():
            if self.strict:
                raise ValueError("imatrix_mse: contains negative values")
            logger.warning(
                "imatrix_mse: contains negative values. Falling back to uniform MSE.",
                log_once=True,
            )
            return None
        if torch.all(imp == 0):
            if self.strict:
                raise ValueError("imatrix_mse: all zeros")
            logger.warning(
                "imatrix_mse: all zeros. Falling back to uniform MSE.", log_once=True
            )
            return None

        if self.args.strategy == QuantizationStrategy.CHANNEL:
            expected = observed.shape[-1]
        elif self.args.strategy in _GROUP_STRATEGIES:
            expected = observed.shape[2] * observed.shape[3]
        else:
            expected = None
        if expected is None:
            if self.strict:
                raise NotImplementedError(
                    f"imatrix_mse: unsupported strategy {self.args.strategy}"
                )
            logger.warning(
                f"imatrix_mse: unsupported strategy {self.args.strategy}."
                " Falling back to uniform MSE.",
                log_once=True,
            )
            return None
        if imp.numel() != expected:
            if self.strict:
                raise ValueError(
                    "imatrix_mse: size mismatch:"
                    f" expected {expected}, got {imp.numel()}"
                )
            logger.warning(
                "imatrix_mse: size mismatch:"
                f" expected {expected}, got {imp.numel()}."
                " Falling back to uniform MSE.",
                log_once=True,
            )
            return None
        return imp


@Observer.register("nvfp4_expanded_imatrix")
class NVFP4ExpandedIMatrixObserver(IMatrixMSEObserver):
    """
    IMatrix observer with defaults tuned for NVFP4 range expansion.

    Same search as :class:`IMatrixMSEObserver` but covers 1.8x down to
    ~0.8x of the per-group range in 112 steps, matching
    :class:`NVFP4ExpandedMSEObserver`.

    Usage::

        QuantizationArgs(
            ...
            observer="nvfp4_expanded_imatrix",
        )
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kw = self.args.observer_kwargs
        if should_generate_mx_scales(self.args):
            self.expand = kw.get("expand", 1.0)
            self.maxshrink = kw.get("maxshrink", 1.0 - 1.0 / 64.0)
        else:
            self.expand = kw.get("expand", 1.8)
            self.maxshrink = kw.get("maxshrink", 1 - 0.8 / 1.8)
        self.grid = kw.get("grid", 200)
        self.norm = kw.get("norm", 2.4)
        self.patience = kw.get("patience", 1000)
