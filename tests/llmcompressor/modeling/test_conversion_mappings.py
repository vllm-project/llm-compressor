import pytest
from compressed_tensors.utils import patch_attr
from transformers.conversion_mapping import (
    USER_REGISTERED_MAPPINGS,
    get_checkpoint_conversion_mapping,
)
from transformers.core_model_loading import WeightRenaming

from llmcompressor.modeling import patch_moe_mappings
from llmcompressor.modeling.moe import conversion_mappings
from llmcompressor.modeling.moe.conversion_mappings import (
    ARCH_TO_2D_MAPPINGS,
    get_conversion_mapping_names,
    get_linearize_load_mappings,
    has_linearize_load_mappings,
    temporary_checkpoint_conversion_mappings,
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


def _mapping_reprs(model_type: str) -> list[str]:
    """Structural snapshot of the cache entry of ``model_type``."""
    entry = get_checkpoint_conversion_mapping(model_type)
    return [repr(transform) for transform in entry]


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


def test_get_conversion_mapping_names_covers_config_tree():
    # sub-models resolve conversion mappings by their own model_type, so the
    # text backbone of a multimodal glm5_next model (glm5_next_text, an alias
    # of the glm5_next pattern) must be registered alongside the top-level
    # model type
    glm5_next = pytest.importorskip("transformers.models.glm5_next")
    config = glm5_next.Glm5NextConfig()

    names = get_conversion_mapping_names("glm5_next", config)

    assert names == ["glm5_next", "glm5_next_text"]
    # the vision tower is not an alias of the pattern and is excluded
    assert "glm5_next_vision" not in names
    # a text-only config covers only its own model type
    assert get_conversion_mapping_names("glm5_next_text", config.text_config) == [
        "glm5_next_text"
    ]


def test_get_conversion_mapping_names_excludes_absent_aliases():
    # only model types present in the config tree being loaded are registered:
    # a qwen2_moe load must not register the pattern's other (cross-family)
    # aliases, such as glm4_moe or qwen3_omni_moe_text
    qwen2_moe = pytest.importorskip("transformers.models.qwen2_moe")

    assert get_conversion_mapping_names("qwen2_moe", qwen2_moe.Qwen2MoeConfig()) == [
        "qwen2_moe"
    ]


def test_text_model_type_loads_linearized():
    # a text-only glm5_next stub resolves its own model_type, which must take
    # the linearized fast path rather than falling back to a post-load
    # conversion of fused 3D experts
    modeling = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

    assert has_linearize_load_mappings("glm5_next_text")

    experts_cls, load_map, _save = get_linearize_load_mappings("glm5_next_text")
    assert experts_cls is modeling.Glm5NextTextExperts

    # the same mapping the multimodal model type resolves
    _experts_cls, multimodal_map, _save = get_linearize_load_mappings("glm5_next")
    assert [repr(transform) for transform in load_map] == [
        repr(transform) for transform in multimodal_map
    ]


def test_temporary_checkpoint_conversion_mappings_restores_previous_entry():
    # glm5_next_text is an alias entry held by transformers' built-in cache
    before = _mapping_reprs("glm5_next_text")
    if not before:
        pytest.skip("transformers version has no glm5_next_text conversion mapping")

    with temporary_checkpoint_conversion_mappings() as register:
        register("glm5_next_text", RENAMINGS)
        assert _mapping_reprs("glm5_next_text") == [repr(r) for r in RENAMINGS]

    # a later load of a 3D-layout glm5_next checkpoint in the same process
    # must see the original (fused) mapping again
    assert _mapping_reprs("glm5_next_text") == before


def test_temporary_checkpoint_conversion_mappings_restores_absent_entry():
    name = "llmcompressor.temporary_checkpoint_conversion_mappings.test"
    assert get_checkpoint_conversion_mapping(name) is None

    with temporary_checkpoint_conversion_mappings() as register:
        register(name, RENAMINGS)
        assert _mapping_reprs(name) == [repr(r) for r in RENAMINGS]
        assert name in USER_REGISTERED_MAPPINGS

    assert get_checkpoint_conversion_mapping(name) is None
    assert name not in USER_REGISTERED_MAPPINGS
