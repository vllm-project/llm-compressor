import pytest
from compressed_tensors.utils import patch_attr
from transformers.core_model_loading import WeightRenaming

from llmcompressor.modeling import patch_moe_mappings
from llmcompressor.modeling.moe import conversion_mappings
from llmcompressor.modeling.moe.conversion_mappings import (
    ARCH_TO_2D_MAPPINGS,
    get_linearize_load_mappings,
    has_linearize_load_mappings,
)

# an architecture whose experts module is registered, but which has no 2D mappings
# and, unlike most architectures, no default conversion mapping either
UNMAPPED_MODEL_TYPE = "qwen3_5_moe"

RENAMINGS = [
    WeightRenaming(
        source_patterns=r"\.experts\.(\d+)\.gate_proj\.",
        target_patterns=r".experts.\1.gate_proj.",
    ),
]


@pytest.fixture
def restore_mappings():
    original = dict(ARCH_TO_2D_MAPPINGS)
    with patch_attr(conversion_mappings, "ARCH_TO_2D_MAPPINGS", original):
        yield


def test_patch_moe_mappings_enables_direct_loading(restore_mappings):
    assert not has_linearize_load_mappings(UNMAPPED_MODEL_TYPE)

    patch_moe_mappings(UNMAPPED_MODEL_TYPE, RENAMINGS)

    assert has_linearize_load_mappings(UNMAPPED_MODEL_TYPE)
    _experts_cls, load_mappings, save_mappings = get_linearize_load_mappings(
        UNMAPPED_MODEL_TYPE
    )
    assert load_mappings == RENAMINGS
    assert save_mappings == RENAMINGS


def test_patch_moe_mappings_removes_targets(restore_mappings):
    remove_targets = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    patch_moe_mappings("qwen2_moe", RENAMINGS, remove_targets=remove_targets)

    _experts_cls, load_mappings, _save_mappings = get_linearize_load_mappings(
        "qwen2_moe"
    )
    assert all(
        target not in remove_targets
        for mapping in load_mappings
        for target in mapping.target_patterns
    )
    assert load_mappings[-len(RENAMINGS) :] == RENAMINGS


def test_nemotron_h_has_linearize_load_mappings():
    assert has_linearize_load_mappings("nemotron_h")


def test_nemotron_h_load_mappings():
    # nemotron_h experts are non-gated: only up_proj and down_proj, no gate_proj
    fused_targets = {"mixer.experts.up_proj", "mixer.experts.down_proj"}

    _experts_cls, load_mappings, save_mappings = get_linearize_load_mappings(
        "nemotron_h"
    )

    # the fused 3D expert targets from the default conversion mapping are dropped
    assert all(
        target not in fused_targets
        for mapping in load_mappings
        for target in mapping.target_patterns
    )

    # per-expert 2D up_proj/down_proj renamings are appended (and nothing gated)
    appended = load_mappings[-2:]
    assert appended == ARCH_TO_2D_MAPPINGS["nemotron_h"][1]
    appended_sources = {source for m in appended for source in m.source_patterns}
    assert appended_sources == {
        r"\.experts\.(\d+)\.up_proj\.",
        r"\.experts\.(\d+)\.down_proj\.",
    }
    assert not any(
        "gate_proj" in target
        for mapping in load_mappings
        for target in mapping.target_patterns
    )

    # the non-expert `backbone.` -> `model.` renaming from the default mapping survives
    assert any("model." in mapping.target_patterns for mapping in load_mappings)

    assert save_mappings == load_mappings


def test_patch_moe_mappings_rejects_unregistered_model_type(restore_mappings):
    with pytest.raises(ValueError, match="no_such_moe"):
        patch_moe_mappings("no_such_moe", RENAMINGS)

    assert not has_linearize_load_mappings("no_such_moe")


def test_patch_moe_mappings_restores_absent_entry(restore_mappings):
    assert not has_linearize_load_mappings(UNMAPPED_MODEL_TYPE)

    with patch_moe_mappings(UNMAPPED_MODEL_TYPE, RENAMINGS):
        assert has_linearize_load_mappings(UNMAPPED_MODEL_TYPE)

    # a checkpoint of this architecture loaded after the block must not be sent down
    # the direct-load path meant for the patched checkpoint
    assert not has_linearize_load_mappings(UNMAPPED_MODEL_TYPE)


def test_patch_moe_mappings_restores_previous_entry(restore_mappings):
    _experts_cls, original_mappings, _save = get_linearize_load_mappings("qwen2_moe")

    with patch_moe_mappings("qwen2_moe", RENAMINGS, remove_targets=["mlp.experts"]):
        _experts_cls, patched_mappings, _save = get_linearize_load_mappings("qwen2_moe")
        assert patched_mappings != original_mappings

    _experts_cls, restored_mappings, _save = get_linearize_load_mappings("qwen2_moe")
    assert restored_mappings == original_mappings
