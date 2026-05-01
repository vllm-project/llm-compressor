from collections import OrderedDict

import pytest
from torch.nn import Linear, Module, Sequential

from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier


class LinearNet(Module):
    _LAYER_DESCS = None

    def __init__(self):
        super().__init__()
        self.seq = Sequential(
            OrderedDict(
                [
                    ("fc1", Linear(8, 16, bias=True)),
                    ("fc2", Linear(16, 32, bias=True)),
                    (
                        "block1",
                        Sequential(
                            OrderedDict(
                                [
                                    ("fc1", Linear(32, 16, bias=True)),
                                    ("fc2", Linear(16, 8, bias=True)),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )


@pytest.mark.unit
def test_smooth_quant_mapping():
    model = LinearNet()
    mappings = [(["seq.fc1"], "seq.fc2")]
    modifier = SmoothQuantModifier(mappings=mappings)

    modifier.ignore = []
    modifier.resolved_mappings_ = modifier._resolve_mappings(model)

    assert len(modifier.resolved_mappings_) == len(mappings)

    mapping = modifier.resolved_mappings_[0]
    assert mapping.smooth_name == mappings[0][1]
    assert isinstance(mapping.smooth_layer, Linear)
    assert isinstance(mapping.balance_layers[0], Linear)
