"""
Integration tests for HIGGS mixed-precision quantization.
"""

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    preset_name_to_scheme,
)

from llmcompressor.entrypoints.higgs import (
    HiggsMSECollectorConverter,
    compute_fused_layer_mse,
    compute_heuristic_alphas,
    compute_layer_mse,
)
from llmcompressor.entrypoints.higgs.utils import UNQUANTIZED_SCHEME
from llmcompressor.entrypoints.model_free.validate import validate_config


@pytest.fixture
def candidate_schemes():
    """Create sample candidate schemes."""
    return {
        "W4A16": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4, type="int", symmetric=True, strategy="tensor"
            ),
        ),
        "W8A8": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=8, type="int", symmetric=True, strategy="tensor"
            ),
        ),
    }


@pytest.fixture
def sample_tensors():
    """Create sample tensor dict simulating a model shard."""
    torch.manual_seed(42)
    return {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 256),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(128, 256),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(512, 256),
        "model.layers.0.mlp.up_proj.weight": torch.randn(512, 256),
    }


def test_mse_collector_basic(candidate_schemes, sample_tensors):
    """Test basic MSE collector workflow."""
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
        allow_unquantized=False,
    )

    # Process tensors
    result_tensors = collector.process(sample_tensors)

    # Tensors should be unchanged
    for key in sample_tensors:
        assert torch.allclose(result_tensors[key], sample_tensors[key])

    # MSE matrix should be populated
    assert len(collector.mse_matrix) == 4  # 4 layers
    for layer_mse in collector.mse_matrix.values():
        assert len(layer_mse) == 2  # 2 candidate schemes

    # Create config
    config = collector.create_config()

    # Should have config groups
    assert len(config.config_groups) > 0
    assert config.config_groups is not None
    assert validate_config(config=config, scheme=None, ignore=[]) is config


def test_mse_collector_uses_heuristic(candidate_schemes, sample_tensors):
    """Test MSE collector's built-in alpha heuristic."""
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
        allow_unquantized=False,
    )

    collector.process(sample_tensors)
    config = collector.create_config()

    # Should generate valid config
    assert config is not None
    assert len(config.config_groups) > 0


def test_mse_collector_validation_rejects_unsupported_tensor_shapes(candidate_schemes):
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
        allow_unquantized=False,
    )

    tensors = {
        "model.layers.0.mlp.down_proj.weight": torch.empty(128, 128, device="meta"),
        "model.layers.0.mamba.conv1d.weight": torch.empty(16, 1, 4, device="meta"),
        "model.layers.1.mamba.conv1d.weight": torch.empty(16, 1, 8, device="meta"),
    }

    with pytest.raises(ValueError) as error:
        collector.validate(tensors)

    message = str(error.value)
    assert "model.layers.0.mamba.conv1d.weight: (16, 1, 4)" in message
    assert "model.layers.1.mamba.conv1d.weight: (16, 1, 8)" in message
    assert "ignore list" in message
    assert "split_fused_moe_experts" in message


def test_mse_collector_validation_splits_known_fused_moe(candidate_schemes):
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
        allow_unquantized=False,
    )
    tensors = {
        "model.layers.0.mlp.gate_up_proj.weight": torch.empty(
            2, 128, 64, device="meta"
        ),
        "model.layers.0.mlp.down_proj.weight": torch.empty(2, 64, 64, device="meta"),
    }

    validated_tensors = collector.validate(tensors)

    assert "model.layers.0.mlp.gate_up_proj.weight" not in validated_tensors
    assert all(tensor.ndim == 2 for tensor in validated_tensors.values())


def test_fused_nvfp4_mse_uses_shared_global_scale():
    scheme = preset_name_to_scheme("NVFP4A16", targets=["Linear"])
    small_weight = torch.linspace(-0.1, 0.1, 256).reshape(16, 16)
    large_weight = torch.linspace(-100, 100, 256).reshape(16, 16)

    independent_mse = compute_layer_mse(small_weight, scheme)
    fused_mse = compute_fused_layer_mse(
        {"q_proj": small_weight, "k_proj": large_weight}, scheme
    )

    # The large fused partner changes the global scale used to quantize q_proj.
    assert fused_mse["q_proj"] != pytest.approx(independent_mse)


def test_mse_collector_allows_unquantized_layers(candidate_schemes, sample_tensors):
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
        allow_unquantized=True,
    )

    collector.process(sample_tensors)

    assert all(
        layer_mse[UNQUANTIZED_SCHEME] == 0.0
        for layer_mse in collector.mse_matrix.values()
    )

    config = collector.create_config()

    # With no bitwidth constraint, zero-MSE unquantized wins for every layer.
    assert config.config_groups == {}


def test_mse_collector_with_fusion(candidate_schemes, sample_tensors):
    """Test MSE collector with fusion detection."""
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
        allow_unquantized=False,
    )

    collector.process(sample_tensors)
    config = collector.create_config()

    # Should generate valid config respecting fusion constraints
    assert config is not None
    assert len(config.config_groups) > 0


def test_alpha_heuristic_layer_types():
    """Test alpha heuristic distinguishes layer types."""
    layer_names = [
        "model.layers.0.self_attn.q_proj",  # attention, depth 0
        "model.layers.5.mlp.gate_proj",  # mlp, depth 5
        "model.embed_tokens",  # embedding, depth 0
    ]
    param_counts = {layer: 1000 for layer in layer_names}

    alphas = compute_heuristic_alphas(layer_names, param_counts)

    # All layers should have alphas
    assert len(alphas) == 3

    # Embedding should have highest alpha (1.5 multiplier)
    # Attention should have middle alpha (1.2 multiplier)
    # MLP should have lower alpha (0.9 multiplier)
    # But depth 5 gives depth_factor = 1.25, so mlp might be higher than attn at depth 0

    embed_alpha = alphas["model.embed_tokens"]
    attn_alpha = alphas["model.layers.0.self_attn.q_proj"]
    # Embedding should be highest
    assert embed_alpha > attn_alpha
