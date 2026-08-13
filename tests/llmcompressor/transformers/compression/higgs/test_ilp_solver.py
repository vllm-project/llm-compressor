"""
Unit tests for ILP solver.
"""

import pytest

from llmcompressor.transformers.compression.higgs.ilp_solver import (
    solve_ilp_mixed_precision,
)


def test_ilp_basic_solution():
    """Verify ILP selects lower MSE scheme when alphas are equal."""
    mse_matrix = {
        "layer1": {"W4A16": 0.5, "W8A8": 0.1},
        "layer2": {"W4A16": 0.3, "W8A8": 0.2},
    }
    alphas = {"layer1": 1.0, "layer2": 1.0}

    solution = solve_ilp_mixed_precision(
        mse_matrix, alphas, ["W4A16", "W8A8"], fused_groups=[]
    )

    # Should choose W8A8 for both layers (lower MSE)
    assert solution["layer1"] == "W8A8"
    assert solution["layer2"] == "W8A8"


def test_ilp_fused_layer_constraint():
    """Verify fused layers get same scheme."""
    mse_matrix = {
        "gate_proj": {"W4A16": 0.1, "W8A8": 0.5},
        "up_proj": {"W4A16": 0.5, "W8A8": 0.1},
    }
    alphas = {"gate_proj": 1.0, "up_proj": 1.0}
    fused_groups = [["gate_proj", "up_proj"]]

    solution = solve_ilp_mixed_precision(
        mse_matrix, alphas, ["W4A16", "W8A8"], fused_groups
    )

    # Both must have same scheme
    assert solution["gate_proj"] == solution["up_proj"]

    # Should choose the scheme that minimizes total cost
    # W4A16: 0.1 + 0.5 = 0.6
    # W8A8: 0.5 + 0.1 = 0.6
    # Either is valid since they're equal
    assert solution["gate_proj"] in ["W4A16", "W8A8"]


def test_ilp_alpha_weighting():
    """Verify alpha weights affect scheme selection."""
    mse_matrix = {
        "important_layer": {"W4A16": 0.5, "W8A8": 0.3},
        "less_important_layer": {"W4A16": 0.1, "W8A8": 0.9},
    }
    # Important layer has high alpha
    alphas = {"important_layer": 10.0, "less_important_layer": 1.0}

    solution = solve_ilp_mixed_precision(
        mse_matrix, alphas, ["W4A16", "W8A8"], fused_groups=[]
    )

    # Important layer should get W8A8 (lower MSE despite higher cost for other layer)
    # Cost with W8A8 for important: 0.3 * 10 = 3.0
    # Cost with W4A16 for important: 0.5 * 10 = 5.0
    # So important layer should get W8A8
    assert solution["important_layer"] == "W8A8"

    # Less important layer should get W4A16 (much lower MSE)
    # Cost: 0.1 * 1 = 0.1 vs 0.9 * 1 = 0.9
    assert solution["less_important_layer"] == "W4A16"


def test_ilp_all_layers_assigned():
    """Verify every layer gets exactly one scheme."""
    mse_matrix = {
        "layer1": {"W4A16": 0.5, "W8A8": 0.3},
        "layer2": {"W4A16": 0.2, "W8A8": 0.4},
        "layer3": {"W4A16": 0.1, "W8A8": 0.2},
    }
    alphas = {"layer1": 1.0, "layer2": 1.0, "layer3": 1.0}

    solution = solve_ilp_mixed_precision(
        mse_matrix, alphas, ["W4A16", "W8A8"], fused_groups=[]
    )

    # All layers should be in solution
    assert set(solution.keys()) == {"layer1", "layer2", "layer3"}

    # Each layer should have a valid scheme
    for layer, scheme in solution.items():
        assert scheme in ["W4A16", "W8A8"]


def test_ilp_multiple_fused_groups():
    """Verify multiple fused groups are handled correctly."""
    mse_matrix = {
        "gate_proj": {"W4A16": 0.1, "W8A8": 0.5},
        "up_proj": {"W4A16": 0.5, "W8A8": 0.1},
        "q_proj": {"W4A16": 0.2, "W8A8": 0.3},
        "k_proj": {"W4A16": 0.3, "W8A8": 0.2},
        "v_proj": {"W4A16": 0.4, "W8A8": 0.1},
    }
    alphas = {layer: 1.0 for layer in mse_matrix}
    fused_groups = [
        ["gate_proj", "up_proj"],
        ["q_proj", "k_proj", "v_proj"],
    ]

    solution = solve_ilp_mixed_precision(
        mse_matrix, alphas, ["W4A16", "W8A8"], fused_groups
    )

    # Group 1: gate_proj and up_proj must match
    assert solution["gate_proj"] == solution["up_proj"]

    # Group 2: q_proj, k_proj, v_proj must all match
    assert solution["q_proj"] == solution["k_proj"]
    assert solution["k_proj"] == solution["v_proj"]


def test_ilp_infinite_mse_excluded():
    """Verify schemes with infinite MSE are excluded."""
    mse_matrix = {
        "layer1": {"W4A16": 0.5, "W8A8": float('inf')},
        "layer2": {"W4A16": 0.3, "W8A8": 0.2},
    }
    alphas = {"layer1": 1.0, "layer2": 1.0}

    solution = solve_ilp_mixed_precision(
        mse_matrix, alphas, ["W4A16", "W8A8"], fused_groups=[]
    )

    # layer1 should get W4A16 (W8A8 has infinite cost)
    assert solution["layer1"] == "W4A16"

    # layer2 should get W8A8 (lower MSE)
    assert solution["layer2"] == "W8A8"


def test_ilp_default_alpha():
    """Verify default alpha of 1.0 is used for missing layers."""
    mse_matrix = {
        "layer1": {"W4A16": 0.5, "W8A8": 0.3},
        "layer2": {"W4A16": 0.2, "W8A8": 0.4},
    }
    # Only provide alpha for layer1
    alphas = {"layer1": 2.0}

    solution = solve_ilp_mixed_precision(
        mse_matrix, alphas, ["W4A16", "W8A8"], fused_groups=[]
    )

    # Should still solve successfully
    assert "layer1" in solution
    assert "layer2" in solution


def test_ilp_empty_fused_groups():
    """Verify empty fused groups don't cause issues."""
    mse_matrix = {
        "layer1": {"W4A16": 0.5, "W8A8": 0.3},
    }
    alphas = {"layer1": 1.0}

    # Empty groups and groups with single layer should be ignored
    fused_groups = [[], ["layer1"]]

    solution = solve_ilp_mixed_precision(
        mse_matrix, alphas, ["W4A16", "W8A8"], fused_groups
    )

    assert solution["layer1"] == "W8A8"  # Lower MSE
