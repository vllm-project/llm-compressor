import pytest
from torch.nn import Linear

from llmcompressor.modifiers.smoothquant import SmoothQuantModifier


@pytest.mark.unit
def test_smooth_quant_mapping(state):
    # Use regex patterns with parent-scoped search
    # ^fc1$ matches only direct child "fc1", not nested "block1.fc1"
    mappings = [(["re:^fc1$"], "re:.*fc2$")]
    modifier = SmoothQuantModifier(mappings=mappings)

    modifier.ignore = []
    modifier.resolved_mappings_ = modifier._resolve_mappings(state.model)

    # Should match seq.fc2 and block1.fc2 (both end with fc2)
    assert len(modifier.resolved_mappings_) == 2

    # Verify seq.fc2 mapping - should find only seq.fc1 (direct child)
    seq_mapping = [
        m for m in modifier.resolved_mappings_ if m.smooth_name == "seq.fc2"
    ][0]
    assert isinstance(seq_mapping.smooth_layer, Linear)
    assert len(seq_mapping.balance_layers) == 1
    assert isinstance(seq_mapping.balance_layers[0], Linear)
