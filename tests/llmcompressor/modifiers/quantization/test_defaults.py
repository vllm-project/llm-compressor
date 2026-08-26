import torch

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.recipe import Recipe

EXPLICIT_FIELDS = {
    "num_bits",
    "type",
    "symmetric",
    "group_size",
    "strategy",
    "block_structure",
    "dynamic",
    "actorder",
    "scale_dtype",
    "zp_dtype",
    "observer",
    "observer_kwargs",
}


def _partial_config_groups():
    return {
        "group_0": {
            "targets": ["Linear"],
            "weights": {"group_size": 128},
        }
    }


def test_direct_modifier_resolves_args_before_nested_validation():
    modifier = QuantizationModifier(config_groups=_partial_config_groups())
    args = modifier.config_groups["group_0"].weights

    assert args.model_fields_set == EXPLICIT_FIELDS
    assert args.num_bits == 8
    assert args.type == "int"
    assert args.symmetric is True
    assert args.strategy == "group"
    assert args.dynamic is False
    assert args.zp_dtype == torch.int8
    assert args.observer == "memoryless_minmax"


def test_recipe_factory_resolves_args_before_nested_validation():
    recipe = Recipe.create_instance(
        """
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      config_groups:
        group_0:
          targets: [Linear]
          weights:
            group_size: 128
"""
    )

    args = recipe.modifiers[0].config_groups["group_0"].weights
    assert args.model_fields_set == EXPLICIT_FIELDS
    assert args.strategy == "group"


def test_kv_cache_args_are_resolved_by_modifier():
    modifier = QuantizationModifier(kv_cache_scheme={"type": "float"})
    args = modifier.kv_cache_scheme

    assert args.model_fields_set == EXPLICIT_FIELDS
    assert args.strategy == "tensor"
    assert args.dynamic is False
    assert args.zp_dtype == torch.float8_e4m3fn


def test_expanded_preset_args_are_fully_resolved():
    modifier = QuantizationModifier(scheme="W4A16")
    args = modifier.resolved_config.config_groups["group_0"].weights

    assert args.model_fields_set == EXPLICIT_FIELDS
    assert args.num_bits == 4
    assert args.strategy == "group"
    assert args.observer == "memoryless_minmax"
