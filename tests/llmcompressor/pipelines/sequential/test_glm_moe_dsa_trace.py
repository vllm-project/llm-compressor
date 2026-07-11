import pytest
import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaForCausalLM,
)

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs
from llmcompressor.utils.dev import skip_weights_initialize


TINY_CONFIG_KWARGS = dict(
    hidden_size=128,
    intermediate_size=256,
    moe_intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    num_local_experts=4,
    n_routed_experts=4,
    num_experts_per_tok=2,
    n_shared_experts=1,
    n_group=1,
    topk_group=1,
    vocab_size=256,
)


@pytest.fixture
def tiny_glm_moe_dsa():
    config = GlmMoeDsaConfig(**TINY_CONFIG_KWARGS)
    with skip_weights_initialize():
        model = GlmMoeDsaForCausalLM(config)
    linearize_moe(model)
    return model


def test_linear_target_fails_trace(tiny_glm_moe_dsa):
    """sequential_targets=['Linear'] must fail: GlmMoeDsaIndexer is untraceable."""
    model = tiny_glm_moe_dsa
    sample_input = {"input_ids": torch.zeros(1, 8, dtype=torch.long)}
    with pytest.raises(Exception):
        trace_subgraphs(
            model,
            sample_input,
            sequential_targets=["Linear"],
            ignore=DatasetArguments().tracing_ignore,
        )


def test_attention_expert_targets_trace(tiny_glm_moe_dsa):
    """sequential_targets=['GlmMoeDsaAttention', 'ExpertMLP'] must trace cleanly."""
    model = tiny_glm_moe_dsa
    sample_input = {"input_ids": torch.zeros(1, 8, dtype=torch.long)}
    subgraphs = trace_subgraphs(
        model,
        sample_input,
        sequential_targets=["GlmMoeDsaAttention", "ExpertMLP"],
        ignore=DatasetArguments().tracing_ignore,
    )
    assert len(subgraphs) > 1
