"""
Tests for ``MoEConfig.from_config`` resolution of architecture-specific MoE
parameters, focusing on nemotron_h's latent MoE.
"""

import pytest
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

from llmcompressor.modeling.moe.helpers import MoEConfig


def _nemotron_config(**overrides) -> NemotronHConfig:
    return NemotronHConfig(
        hidden_size=32,
        moe_intermediate_size=16,
        mlp_hidden_act="relu2",
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        **overrides,
    )


@pytest.mark.unit
def test_moe_config_nemotron_h_latent():
    """A nemotron_h config with a moe_latent_size resolves the experts' width
    into the latent size (while the router keeps scoring the block input), with
    the sigmoid score function its router hardcodes."""
    config = _nemotron_config(moe_latent_size=8)
    moe_config = MoEConfig.from_config(config)

    assert moe_config.num_experts == 8
    assert moe_config.num_experts_per_tok == 2
    assert moe_config.hidden_dim == 8
    assert moe_config.intermediate_size == 16
    assert moe_config.score_function == "sigmoid"
    assert moe_config.hidden_act == "relu2"
    assert moe_config.use_bias is False


@pytest.mark.unit
def test_moe_config_nemotron_h_without_latent():
    """A nemotron_h config without a moe_latent_size runs its experts at the
    hidden size, which both fallbacks resolve to."""
    config = _nemotron_config(moe_latent_size=None)
    moe_config = MoEConfig.from_config(config)

    assert moe_config.hidden_dim == 32
    assert moe_config.score_function == "sigmoid"


@pytest.mark.unit
def test_moe_config_single_group_is_not_group_limited():
    """A config declaring a single routing group (n_group=1, e.g. nemotron_h)
    resolves as ungrouped, so group-limited pruning paths are not taken."""
    config = _nemotron_config(moe_latent_size=8)
    moe_config = MoEConfig.from_config(config)

    assert moe_config.n_group is None
    assert moe_config.top_k_group is None
    assert moe_config.group_size is None
