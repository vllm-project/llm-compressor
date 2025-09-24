from compressed_tensors.utils import deprecated

from llmcompressor.modifiers.pruning.sparsegpt import (
    SparseGPTModifier as PruningSparseGPTModifier,
)

__all__ = ["SparseGPTModifier"]

# Legacy shim for backwards-compat imports


class SparseGPTModifier(PruningSparseGPTModifier):
    @deprecated(
        message=(
            "SparseGPTModifier has moved. In future, please initialize it from "
            "`llmcompressor.modifiers.pruning.sparsegpt.SparseGPTModifier`."
        )
    )
    def __new__(cls, *args, **kwargs):
        return super().__new__(PruningSparseGPTModifier, *args, **kwargs)
