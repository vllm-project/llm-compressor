"""Regression: unit-offset RMSNorm classes must be in the calibration registry."""

import importlib

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from llmcompressor.modeling.offset_norm import (
    NormCalibrationModule,
    norm_calibration_context,
)

# Seven classes that were missing from the alias list (#3429), plus one already
# registered control so a registry regression fails loudly for both groups.
OFFSET_NORM_IMPORTS = [
    ("transformers.models.gemma.modeling_gemma", "GemmaRMSNorm"),
    ("transformers.models.vaultgemma.modeling_vaultgemma", "VaultGemmaRMSNorm"),
    (
        "transformers.models.recurrent_gemma.modeling_recurrent_gemma",
        "RecurrentGemmaRMSNorm",
    ),
    ("transformers.models.t5gemma.modeling_t5gemma", "T5GemmaRMSNorm"),
    ("transformers.models.t5gemma2.modeling_t5gemma2", "T5Gemma2RMSNorm"),
    ("transformers.models.step3p7.modeling_step3p7", "Step3p7RMSNorm"),
    (
        "transformers.models.minimax_m3_vl.modeling_minimax_m3_vl",
        "MiniMaxM3VLRMSNorm",
    ),
    (
        "transformers.models.muse_spark.modeling_muse_spark",
        "MuseSparkTextCenteredRMSNorm",
    ),
]


def _load_norm_cls(module_path: str, class_name: str):
    return getattr(importlib.import_module(module_path), class_name)


@pytest.mark.unit
@pytest.mark.parametrize("module_path,class_name", OFFSET_NORM_IMPORTS)
def test_offset_rmsnorm_registered(module_path: str, class_name: str):
    _load_norm_cls(module_path, class_name)  # ensure class still exists upstream
    lookup_names = set(NormCalibrationModule.registered_names()) | set(
        NormCalibrationModule.registered_aliases()
    )
    # Registry stores standardized (lower/hyphenated) names; accept either form.
    from compressed_tensors.registry import standardize_lookup_name

    assert standardize_lookup_name(class_name) in lookup_names


@pytest.mark.unit
@pytest.mark.parametrize("module_path,class_name", OFFSET_NORM_IMPORTS)
def test_offset_rmsnorm_smoothing_via_calibration_context(
    module_path: str, class_name: str
):
    """Smooth weight inside the context; restored offset-norm output must match
    unsmoothed output divided by the same scales."""
    cls = _load_norm_cls(module_path, class_name)
    dim = 8
    torch.manual_seed(0)
    norm = cls(dim)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(dim) * 0.5)

    class Tiny(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.config = PretrainedConfig()
            self.norm = n

    model = Tiny(norm)
    x = torch.randn(2, 4, dim)
    with torch.no_grad():
        before = model.norm(x).clone()

    scales = torch.linspace(0.5, 1.5, dim)

    with norm_calibration_context(model):
        # Modifiers see standard (1+w) weights and divide them by scales.
        model.norm.weight.data = model.norm.weight.data / scales

    with torch.no_grad():
        after = model.norm(x)

    # Restored offset norm should implement x * (1+w)/s == before / s (broadcast).
    expected = before / scales
    assert torch.allclose(after, expected, atol=1e-5, rtol=1e-5), (
        class_name,
        (after - expected).abs().max().item(),
    )
