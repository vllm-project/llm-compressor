import torch
from transformers import initialization as init
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modeling.moe.helpers import MoEConfig
from llmcompressor.modeling.moe.linear_experts import (
    LinearExperts2D,
    _group_tokens_by_expert,
)


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


def _where_groups(
    top_k_index: torch.Tensor, num_experts: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-expert ``(token_indices, top_k_pos)`` from the old where scan."""
    expert_mask = torch.nn.functional.one_hot(top_k_index, num_experts)
    expert_mask = expert_mask.permute(2, 1, 0)
    groups = []
    for expert_index in range(num_experts):
        top_k_pos, token_indices = torch.where(expert_mask[expert_index])
        groups.append((token_indices, top_k_pos))
    return groups


def _assert_grouping_matches_where(top_k_index: torch.Tensor, num_experts: int):
    token_indices, top_k_pos, counts = _group_tokens_by_expert(top_k_index, num_experts)
    groups = _where_groups(top_k_index, num_experts)

    assert counts == [group[0].numel() for group in groups]
    offset = 0
    for tokens, slots in groups:
        count = tokens.numel()
        assert torch.equal(token_indices[offset : offset + count], tokens)
        assert torch.equal(top_k_pos[offset : offset + count], slots)
        offset += count
    assert offset == token_indices.numel()


def test_group_tokens_by_expert_matches_where_order():
    """Stable sort keeps the slot-major where scan, including duplicate routes."""
    # Token 0 hits expert 0 in every slot, so within-expert order is not a
    # sort by token id. Expert 4 is unused.
    top_k_index = torch.tensor(
        [
            [0, 0, 0],
            [1, 2, 0],
            [2, 1, 1],
            [0, 3, 2],
            [3, 0, 1],
            [1, 1, 2],
            [2, 3, 0],
            [0, 2, 3],
        ]
    )
    _assert_grouping_matches_where(top_k_index, num_experts=5)

    # Routers sometimes emit int32 ids. one_hot rejects those; grouping casts.
    int_tokens, int_slots, int_counts = _group_tokens_by_expert(
        top_k_index.to(torch.int32), num_experts=5
    )
    long_tokens, long_slots, long_counts = _group_tokens_by_expert(
        top_k_index, num_experts=5
    )
    assert int_counts == long_counts
    assert torch.equal(int_tokens, long_tokens)
    assert torch.equal(int_slots, long_slots)

    empty = torch.empty(0, 3, dtype=torch.long)
    tokens, slots, counts = _group_tokens_by_expert(empty, num_experts=5)
    assert tokens.numel() == 0 and slots.numel() == 0
    assert counts == [0, 0, 0, 0, 0]


def _forward_where(
    experts: LinearExperts2D,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    calibrate_all: bool,
) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(hidden_states)
    for expert_index, (token_indices, top_k_pos) in enumerate(
        _where_groups(top_k_index, experts.num_experts)
    ):
        expert = experts[expert_index]
        if calibrate_all:
            expert_output = expert(hidden_states)[token_indices]
        else:
            expert_output = expert(hidden_states[token_indices])
        expert_weights = top_k_weights[token_indices, top_k_pos, None]
        weighted_output = expert_output * expert_weights
        final_hidden_states.index_add_(
            0, token_indices, weighted_output.to(final_hidden_states.dtype)
        )
    return final_hidden_states


def _make_qwen_experts(num_experts: int = 5, top_k: int = 3, seed: int = 0):
    config = Qwen3MoeConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
    )
    experts = LinearExperts2D.get_linear_experts_cls(Qwen3MoeExperts)(config)
    generator = torch.Generator().manual_seed(seed)
    for expert_index in range(experts.num_experts):
        expert = experts[expert_index]
        for projection in (expert.up_proj, expert.gate_proj, expert.down_proj):
            init.normal_(projection.weight, mean=0.0, std=0.02, generator=generator)
    return experts


@torch.no_grad()
def test_linear_experts_forward_matches_where_accumulation():
    """index_add_ order matches the per-expert where scan, in both calib modes."""
    num_tokens = 8
    top_k = 3
    experts = _make_qwen_experts(num_experts=5, top_k=top_k)

    hidden_size = experts[0].up_proj.in_features
    dtype = experts[0].up_proj.weight.dtype
    generator = torch.Generator().manual_seed(1)
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=dtype, generator=generator
    )
    top_k_index = torch.tensor(
        [
            [0, 0, 0],
            [1, 2, 0],
            [2, 1, 1],
            [0, 3, 2],
            [3, 0, 1],
            [1, 1, 2],
            [2, 3, 0],
            [0, 2, 3],
        ]
    )
    top_k_weights = torch.randn(num_tokens, top_k, dtype=dtype, generator=generator)

    routed_inputs: dict[int, torch.Tensor] = {}

    def make_hook(expert_index):
        def hook(module, inputs, output):
            routed_inputs[expert_index] = inputs[0].detach().clone()

        return hook

    hooks = [
        experts[expert_index].register_forward_hook(make_hook(expert_index))
        for expert_index in range(experts.num_experts)
    ]
    try:
        output = experts(hidden_states, top_k_index, top_k_weights)
        # Snapshot before the reference pass, which forwards the same experts.
        seen = {index: value.clone() for index, value in routed_inputs.items()}
        reference = _forward_where(
            experts, hidden_states, top_k_index, top_k_weights, calibrate_all=False
        )
        assert torch.equal(output, reference)

        groups = _where_groups(top_k_index, experts.num_experts)
        for expert_index, (token_indices, _) in enumerate(groups):
            if token_indices.numel() == 0:
                assert expert_index not in seen
                continue
            assert torch.equal(seen[expert_index], hidden_states[token_indices])

        routed_inputs.clear()
        with moe_calibration_context():
            calib_output = experts(hidden_states, top_k_index, top_k_weights)
        calib_seen = {index: value.clone() for index, value in routed_inputs.items()}
        calib_reference = _forward_where(
            experts, hidden_states, top_k_index, top_k_weights, calibrate_all=True
        )
        assert torch.equal(calib_output, calib_reference)
        for expert_index in range(experts.num_experts):
            assert torch.equal(calib_seen[expert_index], hidden_states)
    finally:
        for hook in hooks:
            hook.remove()
