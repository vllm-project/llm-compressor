# ruff: noqa
import warnings

warnings.warn(
    "Importing from llmcompressor.modifiers.quantization.gptq is deprecated. "
    "Please import from llmcompressor.modifiers.gptq instead.",
    DeprecationWarning,
    stacklevel=2,
)

from llmcompressor.modifiers.gptq import *
