"""Compatibility import for recipes using the original modifier name."""

from llmcompressor.modifiers.qad import QADModifier

LayerwiseQADModifier = QADModifier

__all__ = ["LayerwiseQADModifier"]
