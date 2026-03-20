import pytest

from llmcompressor.entrypoints.model_free.microscale import (
    get_fused_names,
    get_unmatched_microscale_names,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "tensor_names,expected_unmatched_count,expected_unmatched_names",
    [
        # QKV: All three present - no unmatched
        (
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ],
            0,
            [],
        ),
        # QKV: Missing v_proj
        (
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
            ],
            2,
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
            ],
        ),
        # QKV: Missing k_proj
        (
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ],
            2,
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ],
        ),
        # gate_up: Both present - no unmatched
        (
            [
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
            ],
            0,
            [],
        ),
        # gate_up: Missing up_proj
        (
            ["model.layers.0.mlp.gate_proj.weight"],
            1,
            ["model.layers.0.mlp.gate_proj.weight"],
        ),
        # gate_up: Missing gate_proj
        (
            ["model.layers.0.mlp.up_proj.weight"],
            1,
            ["model.layers.0.mlp.up_proj.weight"],
        ),
        # w1/w3: Both present - no unmatched
        (
            [
                "model.layers.0.feed_forward.w1.weight",
                "model.layers.0.feed_forward.w3.weight",
            ],
            0,
            [],
        ),
        # w1/w3: Missing w3
        (
            ["model.layers.0.feed_forward.w1.weight"],
            1,
            ["model.layers.0.feed_forward.w1.weight"],
        ),
    ],
)
def test_get_unmatched_microscale_names(
    tensor_names, expected_unmatched_count, expected_unmatched_names
):
    """
    Test that get_unmatched_microscale_names correctly identifies unmatched
    tensors for various fusion patterns (QKV, gate_up, w1/w3).
    """
    unmatched = get_unmatched_microscale_names(tensor_names)
    assert len(unmatched) == expected_unmatched_count
    for expected_name in expected_unmatched_names:
        assert expected_name in unmatched


@pytest.mark.unit
def test_get_unmatched_microscale_names_multiple_layers():
    """
    Test that get_unmatched_microscale_names works correctly with multiple layers,
    some complete and some incomplete.
    """
    tensor_names = [
        # Layer 0: complete QKV
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        # Layer 1: incomplete QKV (missing v_proj)
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.k_proj.weight",
        # Layer 0: complete gate_up
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        # Layer 1: incomplete gate_up (missing up_proj)
        "model.layers.1.mlp.gate_proj.weight",
    ]
    unmatched = get_unmatched_microscale_names(tensor_names)

    # Should have: layer1 q_proj, k_proj, and layer1 gate_proj
    assert len(unmatched) == 3
    assert "model.layers.1.self_attn.q_proj.weight" in unmatched
    assert "model.layers.1.self_attn.k_proj.weight" in unmatched
    assert "model.layers.1.mlp.gate_proj.weight" in unmatched

    # Layer 0 tensors should not be unmatched
    assert "model.layers.0.self_attn.q_proj.weight" not in unmatched
    assert "model.layers.0.mlp.gate_proj.weight" not in unmatched
