"""
Tests for split_fused_moe_experts function.
"""

import torch
from compressed_tensors.utils import match_quantizable_tensors

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


def test_split_fused_moe_experts_direct_parameters_are_quantizable():
    num_experts = 2
    hidden_size = 32
    intermediate_size = 64
    tensors = {
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.float16
        ),
        "model.layers.0.mlp.experts.down_proj": torch.randn(
            num_experts, hidden_size, intermediate_size, dtype=torch.float16
        ),
    }

    result = split_fused_moe_experts(tensors)
    matched_tensor_names = {
        name
        for _, name in match_quantizable_tensors(result, ignore=[], targets=["Linear"])
    }

    expected_tensor_names = {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
    }
    assert set(result.keys()) == expected_tensor_names
    assert matched_tensor_names == expected_tensor_names


def test_split_fused_moe_experts_transposed():
    """Test transposed expert format (e.g. Llama-4) where weights are
    stored as [num_experts, hidden, fused_intermediate] instead of the
    standard [num_experts, fused_intermediate, hidden]."""
    num_experts = 2
    hidden_size = 32
    intermediate_size = 64

    # Standard format: gate_up_proj is [E, 2*intermediate, hidden]
    # Transposed format: gate_up_proj is [E, hidden, 2*intermediate]
    gate_up_standard = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, dtype=torch.float16
    )
    down_standard = torch.randn(
        num_experts, hidden_size, intermediate_size, dtype=torch.float16
    )

    gate_up_transposed = gate_up_standard.transpose(1, 2).contiguous()
    down_transposed = down_standard.transpose(1, 2).contiguous()

    tensors_transposed = {
        "model.layers.0.mlp.experts.gate_up_proj.weight": gate_up_transposed,
        "model.layers.0.mlp.experts.down_proj.weight": down_transposed,
    }

    result = split_fused_moe_experts(tensors_transposed)

    for i in range(num_experts):
        gate_key = f"model.layers.0.mlp.experts.{i}.gate_proj.weight"
        up_key = f"model.layers.0.mlp.experts.{i}.up_proj.weight"
        down_key = f"model.layers.0.mlp.experts.{i}.down_proj.weight"

        assert gate_key in result, f"Missing {gate_key}"
        assert up_key in result, f"Missing {up_key}"
        assert down_key in result, f"Missing {down_key}"

        # All split tensors should be 2D with shape [out_features, in_features]
        assert result[gate_key].shape == (intermediate_size, hidden_size)
        assert result[up_key].shape == (intermediate_size, hidden_size)
        assert result[down_key].shape == (hidden_size, intermediate_size)

    # Verify the values match — transposed split should produce the same
    # result as splitting the standard format
    tensors_standard = {
        "model.layers.0.mlp.experts.gate_up_proj.weight": gate_up_standard,
        "model.layers.0.mlp.experts.down_proj.weight": down_standard,
    }
    result_standard = split_fused_moe_experts(tensors_standard)

    for key in result_standard:
        torch.testing.assert_close(result[key], result_standard[key])
