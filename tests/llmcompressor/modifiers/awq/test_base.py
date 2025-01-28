import pytest

from llmcompressor.modifiers.awq.base import get_layers_in_module
from llmcompressor.transformers import SparseAutoModelForCausalLM


@pytest.mark.parametrize(
    "model_stub, expected_layers",
    [("Xenova/llama2.c-stories15M", 6), ("echarlaix/tiny-random-mistral", 2)],
)
def test_get_model_layers(model_stub, expected_layers):
    """
    Tests if get_model_layers returns the correct number of
    layers in the model
    """

    model = SparseAutoModelForCausalLM.from_pretrained(model_stub)
    layers = get_layers_in_module(model)
    assert len(layers) == expected_layers
