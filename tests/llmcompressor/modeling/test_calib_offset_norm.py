from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llmcompressor.modeling.offset_norm import (
    CalibrationOffsetNorm,
    NormCalibrationModule,
    norm_calibration_context,
)

# ---------------------------------------------------------------------------
# Mock offset-norm module matching Gemma's (1 + weight) convention
# ---------------------------------------------------------------------------


class FakeGemmaRMSNorm(nn.Module):
    """Minimal mock matching the GemmaRMSNorm forward: output * (1 + weight)"""

    def __init__(self, dim, eps=1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=dtype))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


# Patch class name so the registry picks it up
FakeGemmaRMSNorm.__name__ = "GemmaRMSNorm"
FakeGemmaRMSNorm.__qualname__ = "GemmaRMSNorm"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalibrationOffsetNormInit:
    """Test that __init__ converts weights and stores dtype."""

    def test_weight_conversion(self):
        original = FakeGemmaRMSNorm(dim=4)
        original.weight.data = torch.tensor([0.1, -0.05, 0.0, 0.2])
        calib = CalibrationOffsetNorm(original, config=None)

        expected = torch.tensor([1.1, 0.95, 1.0, 1.2])
        assert torch.allclose(calib.weight.data, expected)

    def test_dtype_stored(self):
        original = FakeGemmaRMSNorm(dim=4, dtype=torch.bfloat16)
        calib = CalibrationOffsetNorm(original, config=None)

        assert calib._orig_dtype == torch.bfloat16
        assert calib.weight.dtype == torch.bfloat16


@pytest.mark.unit
class TestCalibrationOffsetNormForward:
    """Test that forward produces the same result as the original."""

    def test_output_matches_original(self):
        original = FakeGemmaRMSNorm(dim=8, dtype=torch.float32)
        original.weight.data = torch.randn(8) * 0.1
        calib = CalibrationOffsetNorm(original, config=None)

        x = torch.randn(2, 4, 8)
        original_out = original(x)
        calib_out = calib(x)

        assert torch.allclose(original_out, calib_out, atol=1e-5)


@pytest.mark.unit
class TestCalibrationOffsetNormRestore:
    """Test that restore reconverts weights correctly."""

    def test_restore_roundtrip(self):
        original = FakeGemmaRMSNorm(dim=4, dtype=torch.bfloat16)
        original.weight.data = torch.tensor(
            [0.1, -0.05, 0.0, 0.2], dtype=torch.bfloat16
        )
        saved = original.weight.data.clone()

        calib = CalibrationOffsetNorm(original, config=None)
        calib.restore(original)

        assert original.weight.dtype == torch.bfloat16
        assert torch.allclose(original.weight.data.float(), saved.float(), atol=2e-2)

    def test_restore_after_smoothing(self):
        original = FakeGemmaRMSNorm(dim=4, dtype=torch.float32)
        original.weight.data = torch.tensor([0.1, -0.05, 0.0, 0.2])

        calib = CalibrationOffsetNorm(original, config=None)
        # Simulate a modifier dividing weights by scales=2
        calib.weight.data.div_(2.0)
        calib.restore(original)

        # Standard weight after smoothing: [1.1, 0.95, 1.0, 1.2] / 2
        #   = [0.55, 0.475, 0.5, 0.6]
        # Restored offset weight: standard - 1
        #   = [-0.45, -0.525, -0.5, -0.4]
        expected = torch.tensor([-0.45, -0.525, -0.5, -0.4])
        assert torch.allclose(original.weight.data, expected, atol=1e-5)

        # Verify: 1 + restored_weight == smoothed standard weight
        effective = 1.0 + original.weight.data
        expected_effective = torch.tensor([0.55, 0.475, 0.5, 0.6])
        assert torch.allclose(effective, expected_effective, atol=1e-5)


@pytest.mark.unit
class TestNormRegistration:
    """Test that registered norms are detected and standard norms are not."""

    def test_gemma_detected(self):
        """GemmaRMSNorm (and aliases) should be in the registry."""
        names = NormCalibrationModule.registered_names()
        aliases = NormCalibrationModule.registered_aliases()
        all_registered = names + aliases
        for name in [
            "gemmarmsnorm",
            "gemma2rmsnorm",
            "gemma3rmsnorm",
            "qwen3nextrmsnorm",
        ]:
            assert name in all_registered, f"{name} not in registry"

    def test_standard_norm_not_detected(self):
        """Standard LayerNorm should not be in the registry."""
        registered = NormCalibrationModule.registered_names()
        assert "layernorm" not in registered
        assert "rmsnorm" not in registered


@pytest.mark.unit
class TestNormCalibrationContext:
    """Test that norm_calibration_context replaces and restores modules."""

    def test_modules_replaced_inside_context(self):
        """Offset norms should be replaced with CalibrationOffsetNorm inside."""
        layer = nn.Module()
        layer.input_layernorm = FakeGemmaRMSNorm(dim=8, dtype=torch.float32)
        layer.post_attention_layernorm = FakeGemmaRMSNorm(dim=8, dtype=torch.float32)

        model = nn.Module()
        model.layer = layer
        model.config = SimpleNamespace(hidden_size=8)

        with norm_calibration_context(model):
            assert isinstance(layer.input_layernorm, CalibrationOffsetNorm)
            assert isinstance(layer.post_attention_layernorm, CalibrationOffsetNorm)

    def test_modules_restored_after_context(self):
        """Original modules should be restored with correct weights."""
        layer = nn.Module()
        layer.input_layernorm = FakeGemmaRMSNorm(dim=4, dtype=torch.bfloat16)
        layer.input_layernorm.weight.data = torch.tensor(
            [0.1, -0.05, 0.0, 0.2], dtype=torch.bfloat16
        )
        saved = layer.input_layernorm.weight.data.clone()

        model = nn.Module()
        model.layer = layer
        model.config = SimpleNamespace(hidden_size=4)

        with norm_calibration_context(model):
            pass

        assert isinstance(layer.input_layernorm, FakeGemmaRMSNorm)
        assert layer.input_layernorm.weight.dtype == torch.bfloat16
        assert torch.allclose(
            layer.input_layernorm.weight.data.float(), saved.float(), atol=2e-2
        )

    def test_weights_updated_after_smoothing(self):
        """Weights modified inside the context should be reflected after."""
        layer = nn.Module()
        layer.norm = FakeGemmaRMSNorm(dim=4, dtype=torch.float32)
        layer.norm.weight.data = torch.tensor([0.1, -0.05, 0.0, 0.2])

        model = nn.Module()
        model.layer = layer
        model.config = SimpleNamespace(hidden_size=4)

        with norm_calibration_context(model):
            # Simulate modifier dividing weights by scales=2
            layer.norm.weight.data.div_(2.0)

        # Standard weight was [1.1, 0.95, 1.0, 1.2] / 2 = [0.55, 0.475, 0.5, 0.6]
        # Restored offset weight: standard - 1 = [-0.45, -0.525, -0.5, -0.4]
        expected = torch.tensor([-0.45, -0.525, -0.5, -0.4])
        assert torch.allclose(layer.norm.weight.data, expected, atol=1e-5)
