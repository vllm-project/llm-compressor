import warnings

from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier

warnings.warn(
    "llmcompressor.modifiers.obcq has been moved to "
    "llmcompressor.modifiers.pruning.sparsegpt Please update your paths",
    DeprecationWarning,
)


__all__ = ["SparseGPTModifier"]
