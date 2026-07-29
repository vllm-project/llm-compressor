import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.conversion_mapping import get_checkpoint_conversion_mapping

from llmcompressor.modeling.kimi_k3 import KimiK3ForConditionalGeneration
from llmcompressor.modeling.kimi_k3.configuration_kimi_k3 import KimiLinearConfig
from llmcompressor.modeling.kimi_k3.modeling_kimi_k3_linear import (
    KimiSparseMoeBlock,
)
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modifiers.pruning.reap.utils import (
    get_moe_attrs,
    prune_moe_layer,
)
from tests.testing_utils import requires_gpu

MODEL_ID = "inference-optimization/Kimi-K3-0.40B"


def _legacy_dispatch(experts, hidden_states, topk_ids, topk_weights):
    outputs = torch.zeros_like(hidden_states)
    expert_mask = torch.nn.functional.one_hot(topk_ids, experts.num_experts).permute(
        2, 1, 0
    )

    for expert_idx, expert in enumerate(experts):
        topk_pos, token_indices = torch.where(expert_mask[expert_idx])
        expert_outputs = expert(hidden_states[token_indices])
        routing_weights = topk_weights[token_indices, topk_pos, None]
        outputs.index_add_(
            0,
            token_indices,
            (expert_outputs * routing_weights).to(outputs.dtype),
        )

    return outputs


def test_kimi_sparse_moe_uses_equivalent_linear_experts():
    torch.manual_seed(0)
    config = KimiLinearConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=None,
        dtype=torch.float32,
    )
    moe = KimiSparseMoeBlock(config)
    with torch.no_grad():
        moe.gate.e_score_correction_bias.zero_()

    hidden_states = torch.randn(2, 3, config.hidden_size)
    topk_ids, topk_weights = moe.gate(hidden_states)
    flattened = hidden_states.flatten(0, 1)

    expected = _legacy_dispatch(moe.experts, flattened, topk_ids, topk_weights)
    actual = moe.experts(flattened, topk_ids, topk_weights)
    moe_attrs = get_moe_attrs(moe, [])

    assert isinstance(moe.experts, LinearExperts2D)
    assert moe_attrs.top_k == config.num_experts_per_token
    assert moe_attrs.n_group == config.num_expert_group
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(moe(hidden_states).flatten(0, 1), expected)

    prune_moe_layer(moe, "", [0, 1, 2], moe_attrs)
    assert isinstance(moe.gate.e_score_correction_bias, torch.nn.Parameter)
    assert moe.gate.e_score_correction_bias.shape == (3,)


@pytest.mark.parametrize(
    ("source_name", "target_name"),
    [
        ("w1", "gate_proj"),
        ("w2", "down_proj"),
        ("w3", "up_proj"),
    ],
)
def test_kimi_expert_weight_mapping_round_trips(source_name, target_name):
    mappings = get_checkpoint_conversion_mapping(KimiLinearConfig.model_type)
    source_key = f"model.layers.1.block_sparse_moe.experts.3.{source_name}.weight"
    target_key = f"model.layers.1.block_sparse_moe.experts.3.{target_name}.weight"

    converted_key = source_key
    for mapping in mappings:
        converted_key, _ = mapping.rename_source_key(converted_key)
    assert converted_key == target_key

    for mapping in reversed(mappings):
        converted_key, _ = mapping.reverse_transform().rename_source_key(converted_key)
    assert converted_key == source_key


@pytest.mark.integration
@requires_gpu
@torch.no_grad()
def test_kimi_k3_checkpoint_loads_with_equivalent_outputs():
    input_ids = torch.tensor([[1, 42, 314, 2]], device="cuda")

    reference = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="cuda",
        dtype=torch.float16,
    )
    expected = reference.language_model(input_ids=input_ids, use_cache=False).logits
    del reference
    torch.cuda.empty_cache()

    model = KimiK3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="cuda",
        dtype=torch.float16,
    )
    actual = model.language_model(input_ids=input_ids, use_cache=False).logits

    moe_layers = [
        layer.block_sparse_moe
        for layer in model.language_model.model.layers
        if hasattr(layer, "block_sparse_moe")
    ]
    assert moe_layers
    assert all(isinstance(layer.experts, LinearExperts2D) for layer in moe_layers)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=2e-2)
