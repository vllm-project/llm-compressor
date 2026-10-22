import pytest
from loguru import logger
from transformers import AutoModelForCausalLM, LlamaConfig

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def _build_model():
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    return AutoModelForCausalLM.from_config(config)


def _recipe():
    return [
        QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    ]


@pytest.mark.unit
def test_oneshot_accepts_a_model_without_a_checkpoint():
    # a model which was created in memory has no checkpoint to read a config
    # from, so oneshot has to fall back to the in-memory config
    model = _build_model()
    assert model.config.name_or_path == ""

    oneshot(model=model, recipe=_recipe())

    assert hasattr(model.model.layers[0].self_attn.q_proj, "quantization_scheme")
    assert not hasattr(model.lm_head, "quantization_scheme")


@pytest.mark.unit
def test_oneshot_warns_for_a_programmatically_quantized_model():
    # the in-memory quantization config must still be picked up, both when it
    # is a dict and when it is an object
    class QuantizationConfig:
        quant_method = "compressed-tensors"

    for quantization_config in (
        {"quant_method": "compressed-tensors"},
        QuantizationConfig(),
    ):
        model = _build_model()
        model.config.quantization_config = quantization_config

        logs = []
        handler_id = logger.add(logs.append, format="{message}", level="WARNING")
        try:
            oneshot(model=model, recipe=_recipe())
        finally:
            logger.remove(handler_id)

        assert any("already quantized" in message for message in logs)
