"""
Unit tests for MSE utilities.
"""

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from llmcompressor.transformers.compression.higgs.mse_utils import (
    compute_layer_mse,
    quantize_weight,
)


@pytest.fixture
def sample_weight():
    """Create a sample weight tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(128, 256)  # [out_features, in_features]


@pytest.fixture
def w8_scheme():
    """8-bit integer quantization scheme."""
    return QuantizationScheme(
        targets=["Linear"],  # Required field
        weights=QuantizationArgs(
            num_bits=8,
            type="int",
            symmetric=True,
            strategy="tensor",
        )
    )


@pytest.fixture
def w4_channel_scheme():
    """4-bit per-channel quantization scheme."""
    return QuantizationScheme(
        targets=["Linear"],  # Required field
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=False,
            strategy="channel",
        )
    )


def test_quantize_weight_tensor_strategy(sample_weight, w8_scheme):
    """Test weight quantization with tensor strategy."""
    quantized = quantize_weight(sample_weight, w8_scheme)

    # Check shape preserved
    assert quantized.shape == sample_weight.shape

    # Check quantized values are different (unless perfect quantization)
    assert not torch.allclose(quantized, sample_weight)

    # Check values are in reasonable range (quantized should be close)
    diff = (quantized - sample_weight).abs()
    assert diff.max() < 1.0  # Max error should be small for 8-bit


def test_quantize_weight_channel_strategy(sample_weight, w4_channel_scheme):
    """Test weight quantization with per-channel strategy."""
    quantized = quantize_weight(sample_weight, w4_channel_scheme)

    # Check shape preserved
    assert quantized.shape == sample_weight.shape

    # Check quantization happened
    assert not torch.allclose(quantized, sample_weight)


def test_compute_layer_mse_returns_scalar(sample_weight, w8_scheme):
    """Test that compute_layer_mse returns a scalar float."""
    mse = compute_layer_mse(sample_weight, w8_scheme)

    assert isinstance(mse, float)
    assert mse >= 0.0  # MSE is always non-negative


def test_compute_layer_mse_higher_bits_lower_error(sample_weight):
    """Test that higher bit quantization has lower MSE."""
    w4_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4, type="int", symmetric=True, strategy="tensor"
        )
    )
    w8_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8, type="int", symmetric=True, strategy="tensor"
        )
    )

    mse_4bit = compute_layer_mse(sample_weight, w4_scheme)
    mse_8bit = compute_layer_mse(sample_weight, w8_scheme)

    # 8-bit should have lower error than 4-bit
    assert mse_8bit < mse_4bit


def test_compute_layer_mse_deterministic(sample_weight, w8_scheme):
    """Test that MSE computation is deterministic."""
    mse1 = compute_layer_mse(sample_weight, w8_scheme)
    mse2 = compute_layer_mse(sample_weight, w8_scheme)

    assert mse1 == mse2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_quantize_weight_on_cuda(sample_weight, w8_scheme):
    """Test quantization on CUDA device."""
    device = torch.device("cuda:0")
    weight_cuda = sample_weight.to(device)

    quantized = quantize_weight(weight_cuda, w8_scheme, device=device)

    assert quantized.device == device
    assert quantized.shape == weight_cuda.shape
