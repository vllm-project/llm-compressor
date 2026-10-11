import pytest

pytest.importorskip("transformers.models.glm_moe_dsa")

import torch  # noqa: E402
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (  # noqa: E402
    GlmMoeDsaConfig,
)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (  # noqa: E402
    GlmMoeDsaAttention,
    GlmMoeDsaForCausalLM,
)

from llmcompressor.args.dataset_arguments import DatasetArguments  # noqa: E402
from llmcompressor.modeling.moe.linearize import linearize_moe  # noqa: E402
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs  # noqa: E402
from llmcompressor.utils.dev import skip_weights_initialize  # noqa: E402

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


def test_linear_targets_trace_with_attention_boundaries(tiny_glm_moe_dsa):
    """Attention boundaries keep the untraceable indexer out of the FX trace."""
    model = tiny_glm_moe_dsa
    sample_input = {"input_ids": torch.zeros(1, 8, dtype=torch.long)}
    subgraphs = trace_subgraphs(
        model,
        sample_input,
        sequential_targets=["Linear"],
        ignore=DatasetArguments().tracing_ignore,
    )
    assert len(subgraphs) > 1
    assert any(
        isinstance(module, GlmMoeDsaAttention)
        for subgraph in subgraphs
        for module in subgraph.submodules(model, recurse=False)
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
