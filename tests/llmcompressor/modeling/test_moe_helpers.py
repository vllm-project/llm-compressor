"""Router score function inference for model types whose config does not declare it."""

import pytest
import torch

from llmcompressor.modeling.moe.helpers import MoEConfig


class _StubConfig:
    """The subset of attributes MoEConfig.from_config reads."""

    num_experts = 8
    num_experts_per_tok = 2
    hidden_size = 32
    intermediate_size = 64
    dtype = torch.float32

    def __init__(self, model_type: str):
        self.model_type = model_type


@pytest.mark.parametrize("model_type", ["glm4_moe", "glm4v_moe_text", "glm4_moe_lite"])
def test_sigmoid_routers_are_detected_without_a_config_field(model_type):
    """transformers applies sigmoid in these routers and the configs do not say so."""
    cfg = MoEConfig.from_config(_StubConfig(model_type))
    assert cfg.score_function == "sigmoid"


def test_declared_score_function_still_wins():
    cfg = _StubConfig("glm4_moe")
    cfg.scoring_func = "softmax"
    assert MoEConfig.from_config(cfg).score_function == "softmax"


@pytest.mark.parametrize("model_type", ["qwen3_5_moe_text", "some_unknown_moe"])
def test_other_model_types_keep_the_softmax_default(model_type):
    cfg = MoEConfig.from_config(_StubConfig(model_type))
    assert cfg.score_function == "softmax"
