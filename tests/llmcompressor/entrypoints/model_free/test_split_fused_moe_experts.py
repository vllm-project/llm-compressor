"""
Tests for split_fused_moe_experts function.
"""

import torch

from llmcompressor.entrypoints.model_free.process import split_fused_moe_experts


def test_split_fused_moe_experts():
    """Test split_fused_moe_experts function."""
    # Construct test input
    num_experts = 2
    hidden_size = 32
    intermediate_size = 64

    tensors = {
        # MoE gate_up_proj: [num_experts, 2*intermediate, hidden]
        "model.layers.0.mlp.experts.gate_up_proj.weight": torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.float16
        ),
        # MoE down_proj: [num_experts, hidden, intermediate]
        "model.layers.0.mlp.experts.down_proj.weight": torch.randn(
            num_experts, hidden_size, intermediate_size, dtype=torch.float16
        ),
        # Non-MoE tensor
        "model.layers.0.self_attn.q_proj.weight": torch.randn(
            32, 32, dtype=torch.float16
        ),
    }

    # Call function
    result = split_fused_moe_experts(tensors)

    # Verify non-MoE tensor remains unchanged
    assert "model.layers.0.self_attn.q_proj.weight" in result
    torch.testing.assert_close(
        result["model.layers.0.self_attn.q_proj.weight"],
        tensors["model.layers.0.self_attn.q_proj.weight"],
    )

    # Verify gate_up_proj is split correctly
    for i in range(num_experts):
        gate_key = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
        up_key = f"model.layers.0.mlp.experts.{i}.up_proj.weight"

        assert gate_key in result
        assert up_key in result
        assert result[gate_key].shape == (intermediate_size, hidden_size)
        assert result[up_key].shape == (intermediate_size, hidden_size)

    # Verify down_proj is split correctly
    for i in range(num_experts):
        down_key = f"model.layers.0.mlp.experts.{i}.down_proj.weight"

        assert down_key in result
        assert result[down_key].shape == (hidden_size, intermediate_size)

    # Verify total tensor count: 1 non-MoE + 2*2 gate_up_proj + 2 down_proj = 7
    assert len(result) == 7
