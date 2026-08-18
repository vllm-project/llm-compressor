import pytest
from transformers.core_model_loading import WeightConverter

from llmcompressor.modeling.moe.conversion_mappings import (
    ARCH_TO_2D_MAPPINGS,
    get_linearize_load_mappings,
    has_linearize_load_mappings,
)

# Qwen3.5-MoE checkpoints store 2D per-expert tensors, so they must take the direct-load
# pathway. Both released spellings are covered: `qwen3_5_moe` is reported by
# the multimodal wrapper config, `qwen3_5_moe_text` by the text-only checkpoint.
QWEN3_5_MOE_TYPES = ["qwen3_5_moe", "qwen3_5_moe_text"]


@pytest.mark.parametrize("model_type", QWEN3_5_MOE_TYPES)
def test_qwen3_5_moe_has_linearize_load_mappings(model_type):
    assert has_linearize_load_mappings(model_type)


@pytest.mark.parametrize("model_type", QWEN3_5_MOE_TYPES)
def test_qwen3_5_moe_load_mappings_avoid_conversion(model_type):
    """
    A remaining `WeightConverter` means weights are fused on load, which is the
    2D -> 3D -> 2D round trip this pathway exists to avoid.
    """
    experts_cls, load_mappings, save_mappings = get_linearize_load_mappings(model_type)

    assert experts_cls is not None
    assert not any(isinstance(mapping, WeightConverter) for mapping in load_mappings)
    assert not any(isinstance(mapping, WeightConverter) for mapping in save_mappings)


@pytest.mark.parametrize("model_type", QWEN3_5_MOE_TYPES)
def test_qwen3_5_moe_load_mappings_keep_expert_renames(model_type):
    """The 2D body must contribute a per-expert rename for each projection."""
    _experts_cls, load_mappings, _save_mappings = get_linearize_load_mappings(
        model_type
    )
    patterns = [
        pattern for mapping in load_mappings for pattern in mapping.source_patterns
    ]

    for projection in ("gate_proj", "up_proj", "down_proj"):
        assert any(projection in pattern for pattern in patterns), projection


@pytest.mark.parametrize("model_type", QWEN3_5_MOE_TYPES)
def test_qwen3_5_moe_keeps_language_model_prefix_rule(model_type):
    """
    Transformers registers Qwen3.5-MoE's rules on the text tower, and they include a
    `model.language_model.*` prefix rule that the Qwen2-MoE rules do not have.
    Resolving the mapping through the wrong spelling would silently drop it.
    """
    _experts_cls, load_mappings, _save_mappings = get_linearize_load_mappings(
        model_type
    )
    patterns = [
        pattern for mapping in load_mappings for pattern in mapping.source_patterns
    ]

    assert any("language_model" in pattern for pattern in patterns)


def test_qwen3_5_moe_reuses_qwen2_moe_2d_body():
    for model_type in ("qwen3_5_moe", "qwen3_5_text"):
        assert ARCH_TO_2D_MAPPINGS[model_type] == ARCH_TO_2D_MAPPINGS["qwen2_moe"]


@pytest.mark.parametrize(
    "model_type", ["qwen2_moe", "qwen3_moe", "qwen3_next", "deepseek_v4", "hy_v3"]
)
def test_existing_architectures_still_resolve(model_type):
    assert has_linearize_load_mappings(model_type)


def test_qwen3_vl_moe_still_uses_post_load_conversion():
    """
    Qwen3-VL-MoE's conversion rules are identity, so its checkpoints are already
    3D and it must keep falling back to `linearize_moe` rather than claiming a
    direct-load pathway.
    """
    assert not has_linearize_load_mappings("qwen3_vl_moe")
