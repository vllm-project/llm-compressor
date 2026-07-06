import torch
from compressed_tensors.quantization import QuantizationStrategy
from compressed_tensors.quantization.lifecycle import fake_quantize
from compressed_tensors.quantization.quant_args import (
    FloatArgs,
    round_to_quantized_type_dtype,
)
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam

from llmcompressor.observers.base import Observer, QParamsDict

__all__ = ["FourOverSixObserver"]

_msg = "Fused module has been garbage collected before its weight was observed"


class _FP8ScaleData256(FloatArgs):
    """FP8 E4M3 with max capped at 256 for FourOverSix global scale.

    Standard NVFP4 uses 448 (full FP8 E4M3 range) as the maximum scale factor.
    FourOverSix uses 256 so that blocks containing a tensor's largest values can
    still select M=4, because 256 * (6/4) = 384 fits in FP8 E4M3.
    """

    exponent = 4
    mantissa = 3
    bits = 8
    max = 256.0
    min = -256.0


@Observer.register("fouroversix")
class FourOverSixObserver(Observer):
    """
    Adaptive block scaling observer for NVFP4 (Four Over Six / 4/6).

    For each block, quantizes with both M=6 (standard NVFP4 full range) and
    M=4 (reduced range that represents near-maximal values more accurately),
    then selects the per-block scale that minimizes quantization error.

    FP4 E2M1 has non-uniform step sizes (0.5 below 2, 1.0 between 2-4, 2.0
    between 4-6), creating large error for values near 5.  Scaling some blocks
    to M=4 makes the representable value distribution more uniform, reducing
    worst-case error for near-maximal values.

    The global scale uses M_FP8=256 instead of 448 so that *every* block,
    including those containing the tensor's largest values, can benefit from
    M=4 (since 256 * 6/4 = 384, which is representable in FP8 E4M3).

    Configurable ``observer_kwargs``:
        scale_selection (str): Error metric for per-block selection.
            "mse"     - mean squared error  (default, best for PTQ)
            "mae"     - mean absolute error (best for pre-training)
            "abs_max" - maximum absolute error

    Reference
    ---------
    Cook et al., "Four Over Six: More Accurate NVFP4 Quantization with
    Adaptive Block Scaling", arXiv:2512.02010, 2025.
    """

    SCALE_EXPANSION = 1.5  # 6 / 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kw = self.args.observer_kwargs or {}
        self.scale_selection: str = kw.get("scale_selection", "mse")
        if self.scale_selection not in ("mse", "mae", "abs_max"):
            raise ValueError(
                f"scale_selection must be 'mse', 'mae', or 'abs_max', "
                f"got '{self.scale_selection}'"
            )
        self._observed_blocks: torch.Tensor | None = None
        self._token_args = self.args.model_copy(
            update={"strategy": QuantizationStrategy.TOKEN}
        )

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        self.min_vals = torch.amin(observed, dim=(0, -1))
        self.max_vals = torch.amax(observed, dim=(0, -1))
        self._observed_blocks = observed.detach().clone()

    @torch.no_grad
    def get_qparams(self) -> QParamsDict:
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
                global_absmax = torch.max(
                    global_absmax, -fused_obs.min_vals.min()
                )
                global_absmax = torch.max(
                    global_absmax, fused_obs.max_vals.max()
                )

            global_scale = generate_gparam(
                -global_absmax.reshape(1),
                global_absmax.reshape(1),
                scale_data=_FP8ScaleData256,
            )

        scale_6, zero_point = calculate_qparams(
            min_vals=self.min_vals,
            max_vals=self.max_vals,
            quantization_args=self.args,
            global_scale=global_scale,
        )

        if self._observed_blocks is not None:
            scale = self._select_block_scales(
                self._observed_blocks, scale_6, zero_point, global_scale
            )
            self._observed_blocks = None
        else:
            scale = scale_6

        return {"scale": scale, "zero_point": zero_point, "global_scale": global_scale}

    # ------------------------------------------------------------------

    def _select_block_scales(
        self,
        observed: torch.Tensor,
        scale_6: torch.Tensor,
        zero_point: torch.Tensor,
        global_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        """Choose M=6 or M=4 per block to minimize quantization error.

        1. Expand M=6 scales by 1.5x → M=4 scales, round to FP8.
        2. Fake-quantize the observed blocks with each scale set.
        3. Pick the scale with lower reconstruction error per block.
        """
        scale_4_raw = scale_6.float() * self.SCALE_EXPANSION
        if self.args.scale_dtype is not None:
            scale_4 = round_to_quantized_type_dtype(
                scale_4_raw, dtype=self.args.scale_dtype
            )
        else:
            scale_4 = scale_4_raw

        q_6 = fake_quantize(
            observed,
            scale_6.unsqueeze(-1),
            zero_point.unsqueeze(-1),
            self._token_args,
            global_scale=global_scale,
        )
        q_4 = fake_quantize(
            observed,
            scale_4.unsqueeze(-1),
            zero_point.unsqueeze(-1),
            self._token_args,
            global_scale=global_scale,
        )

        obs_f = observed.float()
        diff_6 = (q_6.float() - obs_f).abs_()
        diff_4 = (q_4.float() - obs_f).abs_()

        if self.scale_selection == "mse":
            err_6 = diff_6.pow_(2).sum(dim=(0, -1))
            err_4 = diff_4.pow_(2).sum(dim=(0, -1))
        elif self.scale_selection == "mae":
            err_6 = diff_6.sum(dim=(0, -1))
            err_4 = diff_4.sum(dim=(0, -1))
        else:  # abs_max
            err_6 = diff_6.amax(dim=(0, -1))
            err_4 = diff_4.amax(dim=(0, -1))

        select_4 = err_4 < err_6

        # torch.where may not support FP8 directly; operate in float, cast back
        scale_f = torch.where(select_4, scale_4.float(), scale_6.float())
        return scale_f.to(scale_6.dtype)
