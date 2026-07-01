import torch

from llmcompressor.modifiers.quantization import QuantizationModifier


def test_mixfp4a16_recipe_resolves_to_canonical_scheme():
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="MixFP4A16",
        ignore=["lm_head"],
    )

    config = recipe.resolve_quantization_config()
    assert len(config.config_groups) == 1
    scheme = next(iter(config.config_groups.values()))

    assert scheme.targets == ["Linear"]
    assert scheme.format == "mixfp4-pack-quantized"
    assert scheme.weights.num_bits == 4
    assert scheme.weights.type == "float"
    assert scheme.weights.strategy == "tensor_group"
    assert scheme.weights.group_size == 16
    assert scheme.weights.symmetric is True
    assert scheme.weights.dynamic is False
    assert scheme.weights.observer == "mixfp4"
    assert scheme.weights.scale_dtype == torch.float8_e4m3fn
    assert scheme.weights.zp_dtype == torch.float8_e4m3fn
    assert config.ignore == ["lm_head"]
