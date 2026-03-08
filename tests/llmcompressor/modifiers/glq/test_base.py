import pytest

from llmcompressor.modifiers.factory import ModifierFactory


@pytest.mark.unit
@pytest.mark.usefixtures("setup_modifier_factory")
def test_glq_is_registered():
    """Ensure GLQModifier is registered in ModifierFactory"""
    modifier = ModifierFactory.create(
        type_="GLQModifier",
        allow_experimental=False,
        allow_registered=True,
        bits=2,
    )

    from llmcompressor.modifiers.glq import GLQModifier

    assert isinstance(modifier, GLQModifier), "GLQModifier not registered"


@pytest.mark.unit
def test_glq_modifier_defaults():
    """Test GLQModifier default parameters"""
    from llmcompressor.modifiers.glq import GLQModifier

    mod = GLQModifier(bits=2)
    assert mod.bits == 2
    assert mod.targets == "Linear"
    assert mod.ignore == "lm_head"
    assert mod.dampening_frac == 0.01
    assert mod.tune_iters == 0
    assert mod.offload_hessians is False


@pytest.mark.unit
def test_glq_modifier_custom_params():
    """Test GLQModifier with custom parameters"""
    from llmcompressor.modifiers.glq import GLQModifier

    mod = GLQModifier(
        bits=4,
        targets=["Linear"],
        ignore=["lm_head", "embed_tokens"],
        dampening_frac=0.05,
        tune_iters=1,
        offload_hessians=True,
    )
    assert mod.bits == 4
    assert mod.targets == ["Linear"]
    assert mod.ignore == ["lm_head", "embed_tokens"]
    assert mod.dampening_frac == 0.05
    assert mod.tune_iters == 1
    assert mod.offload_hessians is True


@pytest.mark.unit
def test_glq_modifier_bits_values():
    """Test GLQModifier accepts valid bit widths"""
    from llmcompressor.modifiers.glq import GLQModifier

    for bits in (2, 3, 4):
        mod = GLQModifier(bits=bits)
        assert mod.bits == bits
