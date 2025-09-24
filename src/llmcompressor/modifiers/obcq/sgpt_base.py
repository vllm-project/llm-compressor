import warnings
from llmcompressor.modifiers.pruning.sparsegpt.sgpt_base import SparseGPTModifier

__all__ = ["SparseGPTModifier"]

warnings.warn(
    "llmcompressor.modifiers.obcq.sgpt_base is deprecated; "
    "use llmcompressor.modifiers.pruning.sparsegpt.sgpt_base instead.",
    DeprecationWarning,
    stacklevel=2,
)
