# Legacy shim for backwards-compat imports 
import warnings

warnings.warn(
    "llmcompressor.modifiers.obcq is deprecated; "
    "use llmcompressor.modifiers.pruning.sparsegpt instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export public API from the new location
from llmcompressor.modifiers.pruning.sparsegpt import *
