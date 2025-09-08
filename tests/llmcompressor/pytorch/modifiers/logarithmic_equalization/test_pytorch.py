import pytest
from torch.nn import Linear

from llmcompressor.core import State
from llmcompressor.modifiers.logarithmic_equalization import (
    LogarithmicEqualizationModifier,
)
from tests.llmcompressor.pytorch.helpers import LinearNet


@pytest.fixture
def state():
    return State(model=LinearNet())


@pytest.mark.unit
def test_log_equalization_mapping(state):
    mappings = [(["seq.fc2"], "seq.block1.fc1")]
    modifier = LogarithmicEqualizationModifier(mappings=mappings)

    modifier.ignore = []
    modifier.resolved_mappings_ = modifier._resolve_mappings(state.model)

    assert len(modifier.resolved_mappings_) == len(mappings)

    mapping = modifier.resolved_mappings_[0]
    assert mapping.smooth_name == mappings[0][1]
    assert isinstance(mapping.smooth_layer, Linear)
    assert isinstance(mapping.balance_layers[0], Linear)
