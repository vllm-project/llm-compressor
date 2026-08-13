"""
Unit tests for config generator.
"""

import pytest
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from llmcompressor.transformers.compression.higgs.config_generator import (
    create_quantization_config,
    generate_config_groups,
)


@pytest.fixture
def candidate_schemes():
    """Create sample candidate quantization schemes."""
    return {
        "W4A16": QuantizationScheme(
            targets=["Linear"],  # Will be overridden
            weights=QuantizationArgs(
                num_bits=4, type="int", symmetric=True, strategy="channel"
            ),
        ),
        "W8A8": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=8, type="int", symmetric=True, strategy="tensor"
            ),
        ),
    }


def test_generate_config_groups_basic(candidate_schemes):
    """Test basic config_groups generation."""
    ilp_solution = {
        "model.layers.0.q_proj": "W4A16",
        "model.layers.0.k_proj": "W4A16",
        "model.layers.0.mlp.gate_proj": "W8A8",
    }

    config_groups = generate_config_groups(ilp_solution, candidate_schemes)

    # Should have 2 groups (one per unique scheme)
    assert len(config_groups) == 2

    # Check group names
    assert "group_0_W4A16" in config_groups
    assert "group_1_W8A8" in config_groups

    # Check W4A16 group has 2 layers
    w4_scheme = config_groups["group_0_W4A16"]
    assert len(w4_scheme.targets) == 2
    assert any("q_proj" in t for t in w4_scheme.targets)
    assert any("k_proj" in t for t in w4_scheme.targets)

    # Check W8A8 group has 1 layer
    w8_scheme = config_groups["group_1_W8A8"]
    assert len(w8_scheme.targets) == 1
    assert any("gate_proj" in t for t in w8_scheme.targets)


def test_generate_config_groups_regex_escaping(candidate_schemes):
    """Test that layer names are properly escaped in regex patterns."""
    ilp_solution = {
        "model.layers[0].proj": "W4A16",  # Contains special regex chars
    }

    config_groups = generate_config_groups(ilp_solution, candidate_schemes)

    # Should create valid regex pattern
    targets = config_groups["group_0_W4A16"].targets
    assert len(targets) == 1

    # Check that pattern uses re: prefix and escapes special chars
    pattern = targets[0]
    assert pattern.startswith("re:^")
    assert pattern.endswith("$")
    # Dots and brackets should be escaped
    assert "\\." in pattern or "\\[" in pattern


def test_generate_config_groups_all_same_scheme(candidate_schemes):
    """Test when all layers use the same scheme."""
    ilp_solution = {
        "layer1": "W8A8",
        "layer2": "W8A8",
        "layer3": "W8A8",
    }

    config_groups = generate_config_groups(ilp_solution, candidate_schemes)

    # Should have only 1 group
    assert len(config_groups) == 1
    assert "group_0_W8A8" in config_groups

    # Should have all 3 layers
    assert len(config_groups["group_0_W8A8"].targets) == 3


def test_generate_config_groups_preserves_scheme_properties(candidate_schemes):
    """Test that original scheme properties are preserved."""
    ilp_solution = {
        "layer1": "W4A16",
    }

    config_groups = generate_config_groups(ilp_solution, candidate_schemes)
    generated_scheme = config_groups["group_0_W4A16"]

    # Original scheme properties should be preserved
    original_scheme = candidate_schemes["W4A16"]
    assert generated_scheme.weights.num_bits == original_scheme.weights.num_bits
    assert generated_scheme.weights.type == original_scheme.weights.type
    assert generated_scheme.weights.symmetric == original_scheme.weights.symmetric
    assert generated_scheme.weights.strategy == original_scheme.weights.strategy


def test_generate_config_groups_empty_solution(candidate_schemes):
    """Test with empty ILP solution."""
    ilp_solution = {}

    config_groups = generate_config_groups(ilp_solution, candidate_schemes)

    # Should return empty dict
    assert len(config_groups) == 0


def test_generate_config_groups_unknown_scheme(candidate_schemes):
    """Test handling of unknown scheme in solution."""
    ilp_solution = {
        "layer1": "W4A16",
        "layer2": "UNKNOWN_SCHEME",  # Not in candidate_schemes
    }

    config_groups = generate_config_groups(ilp_solution, candidate_schemes)

    # Should only create group for known scheme
    assert len(config_groups) == 1
    # Check that W4A16 group exists (index may vary)
    assert any("W4A16" in name for name in config_groups.keys())
    # Check that layer1 is in the group
    for group in config_groups.values():
        if any("layer1" in t for t in group.targets):
            break
    else:
        pytest.fail("layer1 not found in any config group")


def test_create_quantization_config_basic(candidate_schemes):
    """Test creating QuantizationConfig from config_groups."""
    config_groups = {
        "group_0": candidate_schemes["W4A16"],
        "group_1": candidate_schemes["W8A8"],
    }

    qconfig = create_quantization_config(config_groups, ignore=["lm_head"])

    # Check basic properties
    assert len(qconfig.config_groups) == 2
    assert qconfig.format == "pack-quantized"  # Mixed precision
    assert "lm_head" in qconfig.ignore


def test_create_quantization_config_single_scheme(candidate_schemes):
    """Test with single scheme (not mixed precision)."""
    config_groups = {
        "group_0": candidate_schemes["W4A16"],
    }

    qconfig = create_quantization_config(config_groups)

    # Should still work
    assert len(qconfig.config_groups) == 1
    # Format is still pack-quantized for single group
    assert qconfig.format == "pack-quantized"


def test_generate_config_groups_layer_sorting(candidate_schemes):
    """Test that layers are sorted within each group."""
    ilp_solution = {
        "model.layers.2.proj": "W4A16",
        "model.layers.0.proj": "W4A16",
        "model.layers.1.proj": "W4A16",
    }

    config_groups = generate_config_groups(ilp_solution, candidate_schemes)
    targets = config_groups["group_0_W4A16"].targets

    # Extract layer names from regex patterns
    layer_names = [
        t.replace("re:^", "").replace("$", "").replace("\\.", ".")
        for t in targets
    ]

    # Should be sorted
    assert layer_names == sorted(layer_names)
