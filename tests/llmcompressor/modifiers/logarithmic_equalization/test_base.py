import pytest

from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.logarithmic_equalization.base import (
    LogarithmicEqualizationModifier,
)
from llmcompressor.modifiers.smoothquant.base import SmoothQuantModifier
from tests.llmcompressor.modifiers.conf import setup_modifier_factory


@pytest.mark.unit
def test_logarithmic_equalization_is_registered():
    setup_modifier_factory()

    smoothing_strength = 0.3
    mappings = [(["layer1", "layer2"], "layer3")]
    modifier = ModifierFactory.create(
        type_="LogarithmicEqualizationModifier",
        allow_experimental=False,
        allow_registered=True,
        smoothing_strength=smoothing_strength,
        mappings=mappings,
    )

    assert isinstance(
        modifier, LogarithmicEqualizationModifier
    ), "PyTorch LogarithmicEqualizationModifier not registered"
    assert isinstance(modifier, SmoothQuantModifier)
    assert modifier.smoothing_strength == smoothing_strength
    assert modifier.mappings == mappings
