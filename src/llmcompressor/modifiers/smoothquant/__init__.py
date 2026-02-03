# ruff: noqa
"""
Backwards compatibility shim for SmoothQuantModifier.

This module has been moved to llmcompressor.modifiers.transform.smoothquant.
This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "Importing from 'llmcompressor.modifiers.smoothquant' is deprecated. "
    "Please update your imports to use 'llmcompressor.modifiers.transform.smoothquant' "
    "or 'llmcompressor.modifiers.transform' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

from llmcompressor.modifiers.transform.smoothquant import *
from llmcompressor.modifiers.transform.smoothquant.base import SmoothQuantModifier

__all__ = ["SmoothQuantModifier"]
