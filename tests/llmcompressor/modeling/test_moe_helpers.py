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


@pytest.mark.parametrize(
    "model_type", ["glm4_moe", "glm4_moe_lite", "glm4v_moe", "glm4v_moe_text"]
)
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


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_5_moe_text"])
def test_both_qwen3_5_moe_configs_resolve_to_softmax(model_type):
    """The top-level config and the text sub-config are the same family.

    transformers ships ``Qwen3_5MoeConfig`` (``model_type == "qwen3_5_moe"``) and
    ``Qwen3_5MoeTextConfig`` (``"qwen3_5_moe_text"``), and modifiers see whichever
    one ``get_text_config()`` hands them. Neither declares a router activation, and
    the router applies softmax.
    """
    cfg = MoEConfig.from_config(_StubConfig(model_type))
    assert cfg.score_function == "softmax"


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_5_moe_text"])
def test_inference_does_not_write_back_to_the_config(model_type):
    """Reading the router activation must not add keys to the caller's config.

    The previous form did ``config.scoring_func = "softmax"``, and that key
    survives into any ``config.json`` saved after compression -- a field the
    checkpoint never carried.
    """
    cfg = _StubConfig(model_type)
    before = set(vars(cfg))

    MoEConfig.from_config(cfg)

    added = set(vars(cfg)) - before
    assert set(vars(cfg)) == before, f"from_config added {added} to the config"


def test_known_model_types_are_listed_in_the_router_mapping():
    """The mapping is the contract that keeps the warning from firing for them."""
    from llmcompressor.modeling.moe.helpers import DEFAULT_SCORE_FUNCTION_BY_MODEL_TYPE

    assert DEFAULT_SCORE_FUNCTION_BY_MODEL_TYPE["qwen3_5_moe"] == "softmax"
    assert DEFAULT_SCORE_FUNCTION_BY_MODEL_TYPE["qwen3_5_moe_text"] == "softmax"


@pytest.mark.parametrize("model_type", ["glm_moe_dsa", "glm5_next", "glm5_next_text"])
def test_newer_glm_families_use_sigmoid_too(model_type, caplog):
    """GLM-MoE-DSA and GLM-5 fell into the same gap as GLM-4.

    `GlmMoeDsaTopkRouter` and `Glm5NextTextTopkRouter` call `.sigmoid()`, and
    neither `configuration_glm_moe_dsa.py` nor `configuration_glm5_next.py`
    declares a router activation, so the plain softmax default was wrong for them
    exactly as it was for the GLM-4 family in #3557.
    """
    cfg = MoEConfig.from_config(_StubConfig(model_type))
    assert cfg.score_function == "sigmoid"


def test_glm5_next_vision_is_not_given_a_score_function():
    """The vision tower has no router, so it must not inherit the family's sigmoid."""
    from llmcompressor.modeling.moe.helpers import DEFAULT_SCORE_FUNCTION_BY_MODEL_TYPE

    assert "glm5_next_vision" not in DEFAULT_SCORE_FUNCTION_BY_MODEL_TYPE
