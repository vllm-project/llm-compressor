import pytest
from transformers import AutoModelForCausalLM, LlamaConfig

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.recipe import Recipe


def _config():
    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )


def _model(config, name_or_path):
    model = AutoModelForCausalLM.from_config(config)
    model.config.name_or_path = str(name_or_path)
    return model


@pytest.mark.unit
def test_oneshot_can_reuse_the_same_recipe(tmp_path):
    # a recipe is commonly defined once and reused for several models
    config = _config()
    config.save_pretrained(tmp_path)

    recipe = [
        QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    ]

    for _ in range(2):
        oneshot(model=_model(config, tmp_path), recipe=recipe)


@pytest.mark.unit
def test_oneshot_can_reuse_the_same_recipe_instance(tmp_path):
    config = _config()
    config.save_pretrained(tmp_path)

    recipe = Recipe.create_instance(
        [
            QuantizationModifier(
                targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
            )
        ]
    )

    for _ in range(2):
        oneshot(model=_model(config, tmp_path), recipe=recipe)


@pytest.mark.unit
def test_oneshot_does_not_mutate_the_given_modifiers(tmp_path):
    config = _config()
    config.save_pretrained(tmp_path)

    modifier = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
    )

    oneshot(model=_model(config, tmp_path), recipe=[modifier])

    assert not modifier.initialized
    assert not modifier.finalized
