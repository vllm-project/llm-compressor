from contextlib import nullcontext

import pytest
import torch
from transformers import initialization as init
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modeling.moe.helpers import MoEConfig
from llmcompressor.modeling.moe.linear_experts import ExpertMLP, LinearExperts2D
from llmcompressor.pipelines.sequential.helpers import (
    partition_graph,
    topological_partition,
)


@pytest.mark.parametrize("all_experts", [False, True])
@torch.no_grad()
def test_expert_partitions_preserve_accumulation(all_experts):
    config = Qwen3MoeConfig(hidden_size=16, moe_intermediate_size=32, num_experts=4)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        experts = LinearExperts2D.get_linear_experts_cls(Qwen3MoeExperts)(config)
        for parameter in experts.parameters():
            parameter.normal_(0, 0.1)
        hidden = torch.randn(5, 16)
    indices = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]])
    weights = torch.tensor([[0.7, 0.3]]).expand(5, -1)

    class Tracer(torch.fx.Tracer):
        def is_leaf_module(self, module, name):
            return isinstance(module, ExpertMLP) or super().is_leaf_module(module, name)

    with moe_calibration_context() if all_experts else nullcontext():
        expected = experts(hidden, indices, weights)
        graph = torch.fx.GraphModule(experts, Tracer().trace(experts))
        partitions = partition_graph(
            experts,
            topological_partition(graph, {experts[i] for i in range(4)}),
        )
        assert len(partitions) == 5  # head plus four individual experts
        namespace = dict(
            hidden_states=hidden, top_k_index=indices, top_k_weights=weights
        )
        for partition in partitions:
            # Copy to model cache -> accelerator movement; no alias back to cache.
            inputs = {
                key: namespace[key].clone()
                if isinstance(namespace[key], torch.Tensor)
                else namespace[key]
                for key in partition.input_names
            }
            output = partition.forward(experts, **inputs)
            if isinstance(output, dict):
                namespace.update(output)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


@torch.no_grad()
def test_linear_experts_2d_with_hooks():
    """
    Test LinearExperts2D with forward hooks to verify calibration context behavior.

    This test verifies that:
    1. Outside moe_calibration_context: only selected tokens are sent to each expert
    2. Inside moe_calibration_context: all tokens are sent to all experts
    """
    # Create a Qwen3MoeConfig
    config = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
    )

    # Get the LinearExperts2D class for Qwen3MoeExperts
    linear_experts_cls = LinearExperts2D.get_linear_experts_cls(Qwen3MoeExperts)

    # Create LinearExperts2D instance
    linear_experts = linear_experts_cls(config)

    # Initialize weights to avoid NaN/Inf
    moe_config = MoEConfig.from_config(config)
    for expert_idx in range(linear_experts.num_experts):
        expert = linear_experts[expert_idx]
        init.normal_(expert.up_proj.weight, mean=0.0, std=config.initializer_range)
        init.normal_(expert.gate_proj.weight, mean=0.0, std=config.initializer_range)
        init.normal_(expert.down_proj.weight, mean=0.0, std=config.initializer_range)

    # Create hook counters to track input shapes for each expert
    expert_num_tokens = dict()

    def make_hook(expert_idx):
        def hook(module, input, output):
            num_tokens = input[0].size(0)
            expert_num_tokens[expert_idx] = num_tokens

        return hook

    # Register hooks on each expert
    hooks = []
    for expert_idx in range(linear_experts.num_experts):
        expert = linear_experts[expert_idx]
        hook_handle = expert.register_forward_hook(make_hook(expert_idx))
        hooks.append(hook_handle)

    # Create test inputs
    num_tokens = 16
    hidden_states = torch.randn(
        num_tokens, moe_config.hidden_dim, dtype=moe_config.dtype
    )

    # Create routing: each token goes to 2 experts (top_k=2)
    # Make sure not all tokens go to all experts
    top_k_index = torch.tensor([[0], [1], [2], [3]])
    top_k_weights = torch.randn(
        num_tokens, moe_config.num_experts_per_tok, dtype=moe_config.dtype
    )

    # Test 1: Forward pass WITHOUT calibration context
    expert_num_tokens = dict()
    output_normal = linear_experts(hidden_states, top_k_index, top_k_weights)

    # Verify that not all tokens went to all experts (1 token, see top_k_index)
    for expert_idx in range(moe_config.num_experts):
        input_size = expert_num_tokens[expert_idx]
        assert input_size == 1, (
            f"Without calibration context, expert {expert_idx} should receive "
            f"exactly 1 token, but received {input_size}"
        )

    # Test 2: Forward pass WITH calibration context
    expert_num_tokens = dict()
    with moe_calibration_context():
        output_calib = linear_experts(hidden_states, top_k_index, top_k_weights)

    # Verify that all tokens went to all experts
    for expert_idx in range(moe_config.num_experts):
        input_size = expert_num_tokens[expert_idx]
        assert input_size == num_tokens, (
            f"With calibration context, expert {expert_idx} should receive "
            f"all {num_tokens} tokens, but received {input_size}"
        )

    # Test 3: Verify outputs are valid tensors (not checking exact values since
    # calibration mode changes computation by passing all tokens through experts)
    assert output_normal.shape == hidden_states.shape
    assert output_calib.shape == hidden_states.shape
    assert not torch.isnan(output_normal).any()
    assert not torch.isnan(output_calib).any()

    # Clean up hooks
    for hook in hooks:
        hook.remove()
