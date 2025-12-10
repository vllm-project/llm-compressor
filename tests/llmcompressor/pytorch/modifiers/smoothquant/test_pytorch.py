import pytest
from torch.nn import Linear

from llmcompressor.modifiers.smoothquant import SmoothQuantModifier


@pytest.mark.unit
def test_smooth_quant_mapping(state):
    mappings = [(["seq.fc1"], "seq.fc2")]
    modifier = SmoothQuantModifier(mappings=mappings)

    modifier.ignore = []
    modifier.resolved_mappings_ = modifier._resolve_mappings(state.model)

    assert len(modifier.resolved_mappings_) == len(mappings)

    mapping = modifier.resolved_mappings_[0]
    assert mapping.smooth_name == mappings[0][1]
    assert isinstance(mapping.smooth_layer, Linear)
    assert isinstance(mapping.balance_layers[0], Linear)
