# Legacy shim for backwards-compat imports
import warnings

from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier

__all__ = ["SparseGPTModifier"]


warnings.warn(
    "llmcompressor.modifiers.obcq is deprecated; "
    "use llmcompressor.modifiers.pruning.sparsegpt instead.",
    DeprecationWarning,
    stacklevel=2,
)
