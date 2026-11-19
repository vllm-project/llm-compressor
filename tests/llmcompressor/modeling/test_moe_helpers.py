# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``MoEConfig``."""

import pytest
import torch

from llmcompressor.modeling.moe.helpers import MoEConfig


class _StubConfig:
    """Minimal attribute container matching what ``from_config`` expects of a
    ``PreTrainedConfig``."""

    def __init__(self, **kwargs):
        self.model_type = "stub"
        self.__dict__.update(kwargs)


def test_from_config_reads_routing_fields():
    config = _StubConfig(
        num_experts=16,
        num_experts_per_tok=4,
        n_group=4,
        topk_group=2,
        hidden_size=8,
        moe_intermediate_size=16,
        dtype=torch.bfloat16,
    )
    moe = MoEConfig.from_config(config)
    assert moe.num_experts == 16
    assert moe.num_experts_per_tok == 4
    assert moe.n_group == 4
    assert moe.top_k_group == 2
    assert moe.group_size == 4


def test_from_config_reads_group_routing_aliases():
    # KimiLinear declares group routing under legacy key names
    config = _StubConfig(
        num_experts=8,
        num_experts_per_tok=2,
        num_expert_group=2,
        top_k_group=1,
        hidden_size=8,
        moe_intermediate_size=16,
        dtype=torch.float32,
    )
    moe = MoEConfig.from_config(config)
    assert moe.n_group == 2
    assert moe.top_k_group == 1
    assert moe.group_size == 4


def test_from_config_still_requires_expert_count():
    # the in-memory path keeps its fail-fast behavior for missing fields
    with pytest.raises(AttributeError, match="num_experts"):
        MoEConfig.from_config(_StubConfig(hidden_size=8))


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
    assert moe.num_experts_per_tok == 2
    assert moe.n_group == 2
    assert moe.top_k_group == 1
    assert moe.group_size == 4
    assert moe.num_experts_config_key == "num_experts"


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
    assert moe.num_experts_config_key == "num_local_experts"
    assert moe.num_experts_per_tok == 2
    assert moe.n_group == 2
    assert moe.top_k_group == 1


def test_from_dict_missing_fields_default_to_none():
    moe = MoEConfig.from_dict({"hidden_size": 16})
    assert moe.num_experts is None
    assert moe.num_experts_config_key is None
    assert moe.num_experts_per_tok is None
    assert moe.n_group is None
    assert moe.top_k_group is None
    assert moe.group_size is None


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
    assert moe.intermediate_size == 32
    assert moe.dtype == torch.bfloat16
    assert moe.use_bias is True


def test_from_dict_rejects_indivisible_groups():
    with pytest.raises(ValueError, match="divisible"):
        MoEConfig.from_dict({"num_experts": 7, "n_group": 2})


def test_from_dict_allows_inactive_grouping():
    # n_group=1 (as on GLM / Kimi configs) declares inactive grouping
    moe = MoEConfig.from_dict({"num_experts": 8, "n_group": 1, "topk_group": 1})
    assert moe.group_size == 8
