"""SpinQuant on_initialize keeps user mappings and infers them with transform_config."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.core import State
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.modifiers.transform.spinquant.mappings import (
    infer_mapping_from_model,
)
from llmcompressor.modifiers.transform.spinquant.norm_mappings import NormMapping


def _tiny_llama():
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


def test_transform_config_still_infers_mappings():
    reference = SpinQuantModifier(rotations=["R1", "R2"], transform_type="hadamard")
    reference.on_initialize(State(model=_tiny_llama()))
    config = reference.transform_config

    modifier = SpinQuantModifier(transform_config=config)
    assert modifier.mappings is None
    modifier.on_initialize(State(model=_tiny_llama()))
    assert modifier.mappings is not None
    assert modifier.mappings.embedding is not None
    assert modifier.norm_mappings is not None
    # Must not rebuild / overwrite the provided transform_config.
    assert modifier.transform_config is config


def test_user_mappings_are_preserved():
    model = _tiny_llama()
    mappings = infer_mapping_from_model(model).model_copy(
        update={"mlp_in": ["re:.*up_proj$"]}
    )
    norm_mappings = [
        NormMapping(norm="re:.*custom_norm$", linears=["re:.*custom_linear$"])
    ]
    modifier = SpinQuantModifier(
        rotations=["R1"], mappings=mappings, norm_mappings=norm_mappings
    )
    modifier.on_initialize(State(model=model))
    assert modifier.mappings.mlp_in == ["re:.*up_proj$"]
    assert len(modifier.norm_mappings) == 1
    assert modifier.norm_mappings[0].norm == "re:.*custom_norm$"
    assert modifier.norm_mappings[0].linears == ["re:.*custom_linear$"]
