import pytest
from llmcompressor.modifiers.awq.mappings import get_layer_mappings_from_architecture, AWQMapping

def test_bloom_mappings():
    mappings = get_layer_mappings_from_architecture("BloomForCausalLM")
    # There should be exactly two mappings
    assert len(mappings) == 2
    # Check the first mapping
    m0 = mappings[0]
    assert isinstance(m0, AWQMapping)
    assert m0.smooth_layer == "re:.*input_layernorm$"
    assert m0.balance_layers == ["re:.*self_attention.dense$"]
    # Check the second mapping
    m1 = mappings[1]
    assert isinstance(m1, AWQMapping)
    assert m1.smooth_layer == "re:.*post_attention_layernorm$"
    assert m1.balance_layers == ["re:.*mlp.dense_4h_to_h$"]
