import math
import weakref
from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.utils import patch_attr
from loguru import logger
from torch import distributed as dist

from llmcompressor.observers.base import MinMaxTuple, Observer
from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = ["IMatrixMSEObserver"]

_GROUP_STRATEGIES = (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP)

IMATRIX_PRECISION = torch.float32


@Observer.register("imatrix_mse")
class IMatrixMSEObserver(Observer):
    """
    MSE observer weighted by per-input-channel importance (E[x²]).

    Supports CHANNEL, GROUP, and TENSOR_GROUP for weight-only Linear modules.
    Falls back to uniform MSE when importance data is unavailable.

    Importance is accumulated as raw ``_imatrix_sum`` / ``_imatrix_count``
    and synced across DDP ranks via ``_sync_dict`` before observation.
    """

    _act_sync_dict = {
        "_imatrix_sum": dist.ReduceOp.SUM,
        "_imatrix_count": dist.ReduceOp.SUM,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kw = self.args.observer_kwargs
        self.maxshrink = kw.get("maxshrink", 0.95)
        self.patience = kw.get("patience", 5)
        self.grid = kw.get("grid", 20)
        self.norm = kw.get("norm", 3.0)
        self.strict = kw.get("strict", False)

        self._imatrix_sum: Optional[torch.Tensor] = None
        self._imatrix_count: torch.Tensor = torch.tensor(0, dtype=torch.int64)

        if self.grid <= 0:
            raise ValueError(f"grid must be > 0, got {self.grid}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
        if not (0 <= self.maxshrink <= 1):
            raise ValueError(f"maxshrink must be in [0, 1], got {self.maxshrink}")
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
        """Attach a forward-pre hook to accumulate E[x²] per input channel.

        If raw accumulators (``_imatrix_sum`` / ``_imatrix_count``) already
        exist on the module (second pass after IMatrixGatherer), copy them
        to the observer and skip hook registration.
        """
        self._module_ref = weakref.ref(module)

        if hasattr(module, "_imatrix_sum"):
            self._imatrix_sum = module._imatrix_sum
            self._imatrix_count = module._imatrix_count
            del module._imatrix_sum
            del module._imatrix_count
            return

        if not hasattr(module, "in_features"):
            return

        in_features = module.in_features
        module._imatrix_sum = torch.zeros(in_features, dtype=IMATRIX_PRECISION)
        module._imatrix_count = torch.tensor(0, dtype=torch.int64)

        def _hook(mod, args):
            x = args[0] if isinstance(args, tuple) else args
            if isinstance(x, tuple):
                x = x[0]
            if x is None or not isinstance(x, torch.Tensor):
                return

            x_f = x.detach().to(IMATRIX_PRECISION)
            device = x_f.device
            n_tokens = math.prod(x_f.shape[:-1])
            token_sum = x_f.pow(2).sum(dim=list(range(x_f.dim() - 1)))

            mod._imatrix_sum = mod._imatrix_sum.to(device)
            mod._imatrix_count = mod._imatrix_count.to(device)

            mod._imatrix_sum.add_(token_sum)
            mod._imatrix_count += n_tokens

        module._imatrix_hook = module.register_forward_pre_hook(_hook)

    def detach(self, module: torch.nn.Module) -> None:
        """Remove hooks and leave raw sum/count on module for second-pass pickup.

        Case 1 – accumulators present on module: leave them for next
        observer's ``attach()`` to pick up.

        Case 2 – no accumulators (second-pass cleanup): nothing to do.
        """
        if hasattr(module, "_imatrix_hook"):
            module._imatrix_hook.remove()
            del module._imatrix_hook

    # ------------------------------------------------------------------

    def update_statistics(self, observed: torch.Tensor) -> None:
        importance_weights = self._prepare_importance(observed)
        self.min_vals, self.max_vals = _grid_search(
            observed,
            self.args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            importance_weights=importance_weights,
        )

    # ------------------------------------------------------------------

    def _prepare_importance(self, observed: torch.Tensor) -> Optional[torch.Tensor]:
        """Validate → normalize → broadcast to match observed shape."""
        imp = self._get_validated_importance(observed)
        if imp is None:
            return None

        imp = imp.to(device=observed.device, dtype=torch.float32)
        imp = imp / (imp.mean() + torch.finfo(torch.float32).tiny)

        out_features = observed.shape[1]
        imp_2d = imp.unsqueeze(0).expand(out_features, -1)
        return flatten_for_calibration(imp_2d, self.base_name, self.args)

    def _get_validated_importance(
        self, observed: torch.Tensor
    ) -> Optional[torch.Tensor]:
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


# ---------------------------------------------------------------------------
# TODO: refactor to replace memoryless_mse's grid search, this function
# subsumes it when importance_weights=None.
# ---------------------------------------------------------------------------


def _grid_search(
    observed: torch.Tensor,
    args: QuantizationArgs,
    maxshrink: float,
    patience: int,
    grid: int,
    norm: float,
    importance_weights: Optional[torch.Tensor] = None,
) -> MinMaxTuple:
    """Grid search for min/max minimizing (importance-weighted) quant error.

    Note: global_scale is NOT used during optimization since it cancels out when
    using FP32 scales. After optimization, global_scale is computed from the final
    min/max values in compute_qparams_from_statistics().
    """
    min_val = torch.amin(observed, dim=(0, -1))
    max_val = torch.amax(observed, dim=(0, -1))
    best_error = torch.full(
        min_val.shape,
        torch.finfo(torch.float32).max,
        device=min_val.device,
        dtype=torch.float32,
    )
    best_min = min_val.clone()
    best_max = max_val.clone()

    no_improve = 0
    observed_f = observed.float()

    shrink_steps = max(1, int(maxshrink * grid))
    for i in range(shrink_steps + 1):
        p = 1 - i / grid
        shrink_min = p * min_val
        shrink_max = p * max_val

        scales, zps = calculate_qparams(
            min_vals=shrink_min,
            max_vals=shrink_max,
            quantization_args=args,
            global_scale=None,
        )

        with patch_attr(args, "strategy", QuantizationStrategy.TOKEN):
            q = fake_quantize(
                observed,
                scales.unsqueeze(-1),
                zps.unsqueeze(-1),
                args,
            ).float()

        q.sub_(observed_f).abs_().pow_(norm)
        if importance_weights is not None:
            q.mul_(importance_weights)
        err = q.sum(dim=(0, -1))

        improved = err < best_error
        if torch.any(improved):
            best_error[improved] = err[improved]
            best_min[improved] = shrink_min[improved]
            best_max[improved] = shrink_max[improved]
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                break

    return best_min, best_max
