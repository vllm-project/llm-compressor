"""
Integration tests for HIGGS mixed-precision quantization.
"""

import tempfile

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from llmcompressor.entrypoints.higgs import (
    HiggsMSECollectorConverter,
    HiggsQuantizationConverter,
    compute_heuristic_alphas,
    detect_fused_groups,
)


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


def test_mse_collector_with_heuristic(candidate_schemes, sample_tensors):
    """Test MSE collector with alpha heuristic."""
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
        alpha_calculator=compute_heuristic_alphas,
    )

    collector.process(sample_tensors)
    config = collector.create_config()

    # Should generate valid config
    assert config is not None
    assert len(config.config_groups) > 0


def test_mse_collector_with_fusion(candidate_schemes, sample_tensors):
    """Test MSE collector with fusion detection."""
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
        fusion_detector=detect_fused_groups,
    )

    collector.process(sample_tensors)
    config = collector.create_config()

    # Should generate valid config respecting fusion constraints
    assert config is not None
    assert len(config.config_groups) > 0


def test_quantization_converter_basic(candidate_schemes, sample_tensors):
    """Test quantization converter workflow."""
    # First, run MSE collector to get optimal config
    collector = HiggsMSECollectorConverter(
        candidate_schemes=list(candidate_schemes.values()),
        targets="Linear",
        ignore=[],
    )
    collector.process(sample_tensors)
    optimal_config = collector.create_config()

    # Now apply quantization
    quantizer = HiggsQuantizationConverter(
        optimal_config=optimal_config,
        targets="Linear",
        ignore=[],
    )

    # Process tensors (this quantizes them)
    quantized_tensors = quantizer.process(sample_tensors.copy())

    # Tensors should have changed (quantized)
    # Note: May have different keys due to compression
    assert len(quantized_tensors) > 0


def test_alpha_heuristic_layer_types():
    """Test alpha heuristic distinguishes layer types."""
    layer_names = [
        "model.layers.0.self_attn.q_proj",  # attention, depth 0
        "model.layers.5.mlp.gate_proj",  # mlp, depth 5
        "model.embed_tokens",  # embedding, depth 0
    ]
    param_counts = {
        layer: 1000 for layer in layer_names
    }

    alphas = compute_heuristic_alphas(layer_names, param_counts)

    # All layers should have alphas
    assert len(alphas) == 3

    # Embedding should have highest alpha (1.5 multiplier)
    # Attention should have middle alpha (1.2 multiplier)
    # MLP should have lower alpha (0.9 multiplier)
    # But depth 5 gives depth_factor = 1.25, so mlp might be higher than attn at depth 0

    embed_alpha = alphas["model.embed_tokens"]
    attn_alpha = alphas["model.layers.0.self_attn.q_proj"]
    mlp_alpha = alphas["model.layers.5.mlp.gate_proj"]

    # Embedding should be highest
    assert embed_alpha > attn_alpha


def test_fusion_detection():
    """Test fusion detection identifies fused groups."""
    layer_names = [
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",  # Not fused with gate/up
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]

    groups = detect_fused_groups(layer_names)

    # Should detect 2 groups: (gate_proj, up_proj) and (q_proj, k_proj, v_proj)
    assert len(groups) == 2

    # Find the group containing gate_proj
    gate_group = None
    qkv_group = None

    for group in groups:
        group_str = " ".join(group)
        if "gate_proj" in group_str:
            gate_group = group
        if "q_proj" in group_str:
            qkv_group = group

    # Check gate_proj and up_proj are grouped together
    assert gate_group is not None
    assert len(gate_group) == 2
    assert any("gate_proj" in layer for layer in gate_group)
    assert any("up_proj" in layer for layer in gate_group)

    # Check q/k/v are grouped together
    assert qkv_group is not None
    assert len(qkv_group) == 3
    assert any("q_proj" in layer for layer in qkv_group)
    assert any("k_proj" in layer for layer in qkv_group)
    assert any("v_proj" in layer for layer in qkv_group)
