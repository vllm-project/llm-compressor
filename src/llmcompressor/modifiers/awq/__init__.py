# ruff: noqa
"""
Backwards compatibility shim for AWQModifier.

This module has been moved to llmcompressor.modifiers.transform.awq.
This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "`llmcompressor.modifiers.awq.AWQModifier` is deprecated. "
    "Please update your imports to use 'llmcompressor.modifiers.transform.awq' "
    "or 'llmcompressor.modifiers.transform' instead."
    """
Old API:
    from llmcompressor.modifiers.awq import AWQModifier
    recipe = [
        AWQModifier(
            ignore=["lm_head"],
            scheme="W4A16_ASYM",
            targets=["Linear"],
            duo_scaling="both"
        ),
    ]
New API:
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    recipe = [
        AWQTransformModifier(duo_scaling="both"),
        QuantizationModifier(
            ignore=["lm_head"],
            scheme="W4A16_ASYM",
            targets=["Linear"],
        ),
    ]
This compatibility shim will be removed in a future version.""",
    DeprecationWarning,
    stacklevel=2,
)

from llmcompressor.modifiers.transform.awq import (
    AWQMapping,
    AWQ_MAPPING_REGISTRY,
    AWQ_DYNAMIC_MAPPING_REGISTRY,
    get_layer_mappings_from_model,
    default_mappings,
)

__all__ = [
    "AWQModifier",
    "AWQMapping",
    "AWQ_MAPPING_REGISTRY",
    "AWQ_DYNAMIC_MAPPING_REGISTRY",
    "get_layer_mappings_from_model",
    "default_mappings",
]


def AWQModifier(**kwargs):
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform import AWQModifier as AWQTransformModifier

    quant_keys = (
        "config_groups",
        "targets",
        "ignore",
        "scheme",
        "kv_cache_scheme",
        "weight_observer",
        "input_observer",
        "output_observer",
        "observer",
        "bypass_divisibility_checks",
    )
    quant_kwargs = {k: v for k, v in kwargs.items() if k in quant_keys}
    awq_kwargs = {k: v for k, v in kwargs.items() if k not in quant_keys}

    return [AWQTransformModifier(**awq_kwargs), QuantizationModifier(**quant_kwargs)]
