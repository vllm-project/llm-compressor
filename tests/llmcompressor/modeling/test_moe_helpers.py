# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``MoEConfig``."""

import pytest
import torch
from transformers import PretrainedConfig, Qwen3MoeConfig
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.kimi_linear.configuration_kimi_linear import KimiLinearConfig

from llmcompressor.modeling.moe.helpers import MoEConfig

# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


def test_from_dict_reads_routing_fields():
    moe = MoEConfig.from_dict(
        {
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
        }
    )
    assert moe.num_experts == 8
    assert moe.num_experts_key == "num_experts"
    assert moe.num_experts_per_tok == 2
    assert moe.num_experts_per_tok_key == "num_experts_per_tok"
    assert moe.n_group == 2
    assert moe.n_group_key == "n_group"
    assert moe.top_k_group == 1
    assert moe.top_k_group_key == "topk_group"
    assert moe.group_size == 4


def test_from_dict_reads_text_config():
    # composite (e.g. vision-language) checkpoints keep their MoE fields under
    # text_config
    moe = MoEConfig.from_dict(
        {
            "text_config": {
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "n_group": 2,
                "topk_group": 1,
            }
        }
    )
    assert moe.num_experts == 8
    assert moe.num_experts_per_tok == 2
    assert moe.n_group == 2
    assert moe.top_k_group == 1


def test_from_dict_prefers_top_level_over_text_config():
    moe = MoEConfig.from_dict({"num_experts": 4, "text_config": {"num_experts": 8}})
    assert moe.num_experts == 4


def test_from_dict_reads_key_aliases():
    moe = MoEConfig.from_dict(
        {
            "num_local_experts": 8,
            "top_k_experts": 2,
            "num_expert_group": 2,
            "top_k_group": 1,
        }
    )
    assert moe.num_experts == 8
    assert moe.num_experts_key == "num_local_experts"
    assert moe.num_experts_per_tok == 2
    assert moe.num_experts_per_tok_key == "top_k_experts"
    assert moe.n_group == 2
    assert moe.n_group_key == "num_expert_group"
    assert moe.top_k_group == 1
    assert moe.top_k_group_key == "top_k_group"


def test_from_dict_reads_num_experts_per_token():
    # KimiLinear serializes its per-token selection count under this name
    moe = MoEConfig.from_dict({"num_experts": 8, "num_experts_per_token": 2})
    assert moe.num_experts_per_tok == 2
    assert moe.num_experts_per_tok_key == "num_experts_per_token"


def test_from_dict_missing_fields_default_to_none():
    moe = MoEConfig.from_dict({"hidden_size": 16})
    for field in (
        "num_experts",
        "num_experts_per_tok",
        "n_group",
        "top_k_group",
        "group_size",
    ):
        assert getattr(moe, field) is None
    for field in ("num_experts", "num_experts_per_tok", "n_group", "top_k_group"):
        assert getattr(moe, f"{field}_key") is None
    assert moe.use_bias is False


def test_from_dict_skips_group_size_without_num_experts():
    # a checkpoint config may declare group routing without an expert count
    # (e.g. when the count is only derivable from the weights); group_size is
    # left to the consumer, which knows the true count
    moe = MoEConfig.from_dict({"n_group": 2, "topk_group": 1})
    assert moe.group_size is None


def test_from_dict_reads_shape_fields():
    moe = MoEConfig.from_dict(
        {
            "hidden_size": 16,
            "moe_intermediate_size": 32,
            "torch_dtype": "bfloat16",
            "mlp_bias": True,
        }
    )
    assert moe.hidden_dim == 16
    assert moe.hidden_dim_key == "hidden_size"
    assert moe.intermediate_size == 32
    assert moe.intermediate_size_key == "moe_intermediate_size"
    assert moe.dtype == torch.bfloat16
    assert moe.dtype_key == "torch_dtype"
    assert moe.use_bias is True
    assert moe.use_bias_key == "mlp_bias"


@pytest.mark.parametrize("dtype_name", ["bfloat16", "torch.bfloat16"])
def test_from_dict_converts_dtype_names(dtype_name):
    moe = MoEConfig.from_dict({"dtype": dtype_name})
    assert moe.dtype == torch.bfloat16
    assert moe.dtype_key == "dtype"


def test_from_dict_rejects_indivisible_groups():
    with pytest.raises(ValueError, match="divisible"):
        MoEConfig.from_dict({"num_experts": 7, "n_group": 2})


def test_from_dict_allows_inactive_grouping():
    # n_group=1 (as on GLM / Kimi configs) declares inactive grouping
    moe = MoEConfig.from_dict({"num_experts": 8, "n_group": 1, "topk_group": 1})
    assert moe.group_size == 8


def test_from_dict_applies_gpt_oss_special_case():
    moe = MoEConfig.from_dict({"model_type": "gpt_oss", "num_experts": 8})
    assert moe.use_bias is True
    assert moe.limit == 7.0
    assert moe.alpha == 1.702
    assert moe.hidden_act == "sigmoid"


def test_from_dict_applies_lfm2_moe_special_case():
    moe = MoEConfig.from_dict({"model_type": "lfm2_moe"})
    assert moe.hidden_act == "silu"


def test_from_dict_nemotron_h_falls_back_to_hidden_size():
    # moe_latent_size is declared but unset: the expert hidden dim is the
    # model hidden size
    moe = MoEConfig.from_dict(
        {"model_type": "nemotron_h", "moe_latent_size": None, "hidden_size": 512}
    )
    assert moe.hidden_dim == 512
    assert moe.hidden_dim_key == "hidden_size"


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def test_from_config_delegates_to_from_dict():
    config = KimiLinearConfig(
        hidden_size=16, intermediate_size=32, num_experts=8, num_experts_per_tok=2
    )
    assert MoEConfig.from_config(config) == MoEConfig.from_dict(config.to_dict())


def test_from_config_reads_canonical_key_names():
    # a PreTrainedConfig serializes under canonical key names, which may
    # differ from its attribute names
    config = Qwen3MoeConfig(
        hidden_size=16, intermediate_size=32, num_experts=4, num_experts_per_tok=2
    )
    moe = MoEConfig.from_config(config)
    assert moe.num_experts == 4
    assert moe.num_experts_key == "num_local_experts"
    assert moe.num_experts_per_tok == 2
    assert moe.hidden_dim == 16
    assert moe.hidden_act == "silu"

    # KimiLinear's per-token selection count serializes as
    # num_experts_per_token
    config = KimiLinearConfig(
        hidden_size=16, intermediate_size=32, num_experts=8, num_experts_per_tok=2
    )
    moe = MoEConfig.from_config(config)
    assert moe.num_experts == 8
    assert moe.num_experts_per_tok == 2
    assert moe.num_experts_per_tok_key == "num_experts_per_token"

    # DeepseekV3 declares its experts under n_routed_experts and carries
    # group-limited routing
    config = DeepseekV3Config(
        hidden_size=16,
        moe_intermediate_size=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
    )
    moe = MoEConfig.from_config(config)
    assert moe.num_experts == 4
    assert moe.num_experts_key == "n_routed_experts"
    assert moe.n_group == 2
    assert moe.top_k_group == 1
    assert moe.group_size == 2


def test_from_config_missing_fields_default_to_none():
    # from_config standardizes on from_dict and is equally lenient
    moe = MoEConfig.from_config(PretrainedConfig())
    assert moe.num_experts is None
    assert moe.num_experts_key is None
    assert moe.num_experts_per_tok is None
