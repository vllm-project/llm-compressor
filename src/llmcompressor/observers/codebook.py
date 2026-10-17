"""
Codebook-selection observers for quantization calibration.

MixFP4 chooses NVFP4 E2M1 or signed INT4 per group and stores the choice in
unused bit 7 of the FP8 per-group scale byte:
  bit7 = 0 -> FP4 E2M1 group
  bit7 = 1 -> signed INT4 group
"""

import torch
from compressed_tensors.quantization import (
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from llmcompressor.observers.base import MinMaxTuple, Observer, ScaleZpTuple
from llmcompressor.observers.helpers import flatten_for_calibration

__all__ = ["MixFP4Observer"]

_FP4_MAX = 6.0
_INT4_MAX = 7.0
_FP8_MAX = FP8_E4M3_DATA.max
_E2M1_CODEBOOK = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


@Observer.register("mixfp4", alias="mixed_fp4_int4")
class MixFP4Observer(Observer):
    """
    Observer that chooses FP4 E2M1 or signed INT4 per tensor-group by MSE.

    MixFP4 uses the same 4-bit packed weights and FP8 per-group scale storage
    as NVFP4. The INT4/FP4 selector is packed into the FP8 scale sign bit, so
    no extra metadata is produced.
    """

    def __init__(
        self,
        base_name: str,
        args: QuantizationArgs,
        module: torch.nn.Module | None = None,
        **observer_kwargs,
    ):
        super().__init__(
            base_name=base_name, args=args, module=module, **observer_kwargs
        )
        self._validate_args()

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_min_max(observed)

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_min_max(observed)

    @torch.no_grad()
    def forward(self, observed: torch.Tensor) -> ScaleZpTuple:
        g_idx = self._get_module_param("g_idx")
        global_scale = self._get_module_param("global_scale")
        self._check_has_global_scale(global_scale)

        observed = flatten_for_calibration(observed, self.base_name, self.args, g_idx)
        scales = self._calculate_mixfp4_scales(observed, global_scale)
        zero_points = torch.zeros_like(scales)
        return scales, zero_points

    def _calculate_mixfp4_scales(
        self, observed: torch.Tensor, global_scale: torch.Tensor
    ) -> torch.Tensor:
        global_scale = global_scale.to(torch.float32).reshape(-1)[0].to(observed.device)
        max_abs_group = observed.abs().amax(dim=(0, -1))

        mag_fp4 = _quantize_fp8_magnitude(global_scale * max_abs_group / _FP4_MAX)
        mag_int4 = _quantize_fp8_magnitude(global_scale * max_abs_group / _INT4_MAX)

        min_scale = torch.finfo(torch.float32).tiny
        eff_fp4 = (mag_fp4 / global_scale).clamp(min=min_scale)
        eff_int4 = (mag_int4 / global_scale).clamp(min=min_scale)

        err_fp4 = _fake_quant_error_fp4(observed, eff_fp4)
        err_int4 = _fake_quant_error_int4(observed, eff_int4)
        is_int4 = err_int4 < err_fp4
        magnitudes = torch.where(is_int4, mag_int4, mag_fp4)

        return _pack_scale_flag(magnitudes, is_int4)

    def _validate_args(self) -> None:
        if self.base_name != "weight":
            raise ValueError("MixFP4Observer only supports weight quantization")
        if self.args.strategy != QuantizationStrategy.TENSOR_GROUP:
            raise ValueError("MixFP4Observer requires tensor_group strategy")
        if self.args.group_size != 16:
            raise ValueError("MixFP4Observer requires group_size=16")
        if self.args.num_bits != 4:
            raise ValueError("MixFP4Observer requires num_bits=4")
        if self.args.type != QuantizationType.FLOAT:
            raise ValueError("MixFP4Observer requires floating-point 4-bit weights")
        if not self.args.symmetric:
            raise ValueError("MixFP4Observer requires symmetric quantization")
        if self.args.scale_dtype != torch.float8_e4m3fn:
            raise ValueError("MixFP4Observer requires float8_e4m3fn scales")


def _get_min_max(observed: torch.Tensor) -> MinMaxTuple:
    return torch.amin(observed, dim=(0, -1)), torch.amax(observed, dim=(0, -1))


def _quantize_fp8_magnitude(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(min=0, max=_FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def _fake_quant_error_fp4(observed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    codebook = _E2M1_CODEBOOK.to(device=observed.device)
    scale = scale.unsqueeze(0).unsqueeze(-1)
    scaled = observed / scale
    sign = torch.sign(scaled)
    magnitude = scaled.abs().clamp(max=_FP4_MAX)
    indices = (magnitude.unsqueeze(-1) - codebook).abs().argmin(dim=-1)
    dequantized = sign * codebook[indices] * scale
    return ((dequantized - observed) ** 2).sum(dim=(0, -1))


def _fake_quant_error_int4(observed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.unsqueeze(0).unsqueeze(-1)
    quantized = (observed / scale).round().clamp(-_INT4_MAX, _INT4_MAX)
    dequantized = quantized * scale
    return ((dequantized - observed) ** 2).sum(dim=(0, -1))


def _pack_scale_flag(magnitude: torch.Tensor, is_int4: torch.Tensor) -> torch.Tensor:
    magnitude_fp8 = magnitude.clamp(min=0, max=_FP8_MAX).to(torch.float8_e4m3fn)
    raw = magnitude_fp8.contiguous().view(torch.uint8)
    nonzero_magnitude = (raw & 0x7F) != 0
    sign_mask = ((is_int4 & nonzero_magnitude).to(torch.uint8) & 1) << 7
    packed = (raw & 0x7F) | sign_mask
    return packed.view(torch.float8_e4m3fn)
