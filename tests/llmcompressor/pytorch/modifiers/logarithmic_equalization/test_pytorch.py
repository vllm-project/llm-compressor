import pytest
from torch.nn import Linear

from llmcompressor.modifiers.logarithmic_equalization import (
    LogarithmicEqualizationModifier,
)


@pytest.mark.unit
def test_log_equalization_mapping(state):
    # Use regex patterns with parent-scoped search
    # Searches for balance layers within the parent of smooth layer
    mappings = [(["re:^fc2$"], "re:.*block1\\.fc1$")]
    modifier = LogarithmicEqualizationModifier(mappings=mappings)

    modifier.ignore = []
    modifier.resolved_mappings_ = modifier._resolve_mappings(state.model)

    assert len(modifier.resolved_mappings_) == 1

    mapping = modifier.resolved_mappings_[0]
    assert mapping.smooth_name == "seq.block1.fc1"
    assert isinstance(mapping.smooth_layer, Linear)
    assert len(mapping.balance_layers) == 1
    assert isinstance(mapping.balance_layers[0], Linear)
