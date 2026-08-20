import torch
from compressed_tensors.quantization import QuantizationStrategy
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam

from llmcompressor.observers.base import Observer, QParamsDict
from llmcompressor.observers.mse_quant import _grid_search_mse

__all__ = ["NVFP4TwoPassObserver"]


@Observer.register("nvfp4_twopass")
class NVFP4TwoPassObserver(Observer):
    """Two-pass grid search observer for NVFP4 quantization.

    Pass 1 (during ``forward``): Expanded MSE grid search with no
    global_scale and no FP8 rounding (scale_dtype=None).  This is
    the "perfect search" oracle — scales are optimized in float32.
    All fused observers complete pass 1 before any starts pass 2.

    Pass 2 (during ``get_qparams``): Re-run the grid search with the
    fused global_scale derived from pass-1 ranges, using the real
    scale_dtype so the error accounts for FP8 rounding.
    """

    _act_sync_dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.strategy != QuantizationStrategy.TENSOR_GROUP:
            raise ValueError(
                "nvfp4_twopass observer requires TENSOR_GROUP strategy "
                f"(got {self.args.strategy})"
            )
        observer_kwargs = self.args.observer_kwargs
        self.maxshrink = observer_kwargs.get("maxshrink", 1 - 0.8 / 1.8)
        self.patience = observer_kwargs.get("patience", 1000)
        self.grid = observer_kwargs.get("grid", 200.0)
        self.norm = observer_kwargs.get("norm", 2.4)
        self.chunk_size = observer_kwargs.get("chunk_size", 5)
        self.expand = observer_kwargs.get("expand", 1.8)
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        self._token_args = self.args.model_copy(
            update={"strategy": QuantizationStrategy.TOKEN}
        )
        # Pass 1 uses float32 scales (no FP8 rounding)
        self._nofp8_args = self.args.model_copy(
            update={"scale_dtype": None}
        )
        self._nofp8_token_args = self._nofp8_args.model_copy(
            update={"strategy": QuantizationStrategy.TOKEN}
        )

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        self._observed = observed.to("cpu", non_blocking=True)
        self._observed_device = observed.device
        self.min_vals, self.max_vals = _grid_search_mse(
            observed,
            self._nofp8_args,
            self._nofp8_token_args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm,
            self.chunk_size,
            self.expand,
            global_scale=None,
        )
        self._pass1_min_vals = self.min_vals.clone()
        self._pass1_max_vals = self.max_vals.clone()

    def _fused_global_scale(self) -> torch.Tensor:
        """Compute global_scale from pass-1 ranges of all fused observers."""
        absmax = torch.max(-self._pass1_min_vals.min(), self._pass1_max_vals.max())
        for handler in self.fusion_handler._group:
            obs = handler._observer
            p1_min = obs._pass1_min_vals
            p1_max = obs._pass1_max_vals
            absmax = torch.max(absmax, -p1_min.min())
            absmax = torch.max(absmax, p1_max.max())

        absmax = torch.clamp(absmax, min=torch.finfo(absmax.dtype).tiny)
        return generate_gparam(-absmax.reshape(1), absmax.reshape(1))

    @torch.no_grad
    def get_qparams(self) -> QParamsDict:
        assert self.has_statistics, (
            "No statistics available. Call observer(value) first."
        )

        global_scale = self._fused_global_scale()

        observed = self._observed.to(self._observed_device)
        del self._observed
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
            global_scale=global_scale,
        )
        del observed

        scale, zero_point = calculate_qparams(
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            quantization_args=self.args,
            global_scale=global_scale,
        )
        self.delete_statistics()

        return {"scale": scale, "zero_point": zero_point, "global_scale": global_scale}
