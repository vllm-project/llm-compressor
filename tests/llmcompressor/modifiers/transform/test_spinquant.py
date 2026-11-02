import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.core import State
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.modifiers.transform.spinquant.mappings import (
    infer_mapping_from_model,
)


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    return LlamaForCausalLM(config).eval()


def _apply(modifier: SpinQuantModifier, model: torch.nn.Module) -> torch.Tensor:
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, None)
    with torch.no_grad():
        return model(torch.arange(16).unsqueeze(0)).logits


@pytest.mark.unit
def test_reloaded_modifier_with_transform_config():
    modifier = SpinQuantModifier(rotations=["R1", "R2"], transform_type="hadamard")
    expected = _apply(modifier, _tiny_llama())

    # the serialized modifier keeps `transform_config` but not the mappings
    reloaded = SpinQuantModifier.model_validate(modifier.model_dump())
    assert reloaded.transform_config is not None
    assert reloaded.mappings is None

    torch.testing.assert_close(_apply(reloaded, _tiny_llama()), expected)


@pytest.mark.unit
def test_user_mappings_are_kept():
    model = _tiny_llama()
    mappings = infer_mapping_from_model(model).model_copy(
        update={"mlp_in": ["re:.*up_proj$"]}
    )
    modifier = SpinQuantModifier(rotations=["R1"], mappings=mappings)
    modifier.on_initialize(State(model=model))

    assert modifier.mappings.mlp_in == ["re:.*up_proj$"]
