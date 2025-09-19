import pytest

from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.smoothquant.base import SmoothQuantModifier


@pytest.mark.unit
@pytest.mark.usefixtures("setup_modifier_factory")
def test_smooth_quant_is_registered():
    smoothing_strength = 0.3
    mappings = [(["layer1", "layer2"], "layer3")]
    modifier = ModifierFactory.create(
        type_="SmoothQuantModifier",
        allow_experimental=False,
        allow_registered=True,
        smoothing_strength=smoothing_strength,
        mappings=mappings,
    )

    assert isinstance(
        modifier, SmoothQuantModifier
    ), "PyTorch SmoothQuant not registered"
    assert modifier.smoothing_strength == smoothing_strength
    assert modifier.mappings == mappings


@pytest.mark.unit
@pytest.mark.usefixtures("setup_modifier_factory")
def test_smooth_quant_defaults():
    default_sq = SmoothQuantModifier()
    assert default_sq.smoothing_strength == 0.5


@pytest.mark.unit
def test_override_defaults():
    strength = 0.7
    dummy_map = [(["layer1", "layer2"], "layer3")]
    non_default_sq = SmoothQuantModifier(
        smoothing_strength=strength, mappings=dummy_map
    )

    assert non_default_sq.smoothing_strength == strength
    assert non_default_sq.mappings == dummy_map
