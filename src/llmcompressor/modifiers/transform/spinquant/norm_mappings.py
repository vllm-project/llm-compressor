"""Compatibility imports for norm fusion mappings."""

from llmcompressor.modeling.norm_mappings import (
    NORM_MAPPING_REGISTRY,
    NormMapping,
    infer_norm_mapping_from_model,
)

__all__ = ["NORM_MAPPING_REGISTRY", "NormMapping", "infer_norm_mapping_from_model"]
