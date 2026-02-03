# ruff: noqa
"""
Backwards compatibility shim for SmoothQuantModifier base module.

This module has been moved to llmcompressor.modifiers.transform.smoothquant.base.
This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "Importing from 'llmcompressor.modifiers.smoothquant.base' is deprecated. "
    "Please update your imports to use 'llmcompressor.modifiers.transform.smoothquant.base' "
    "instead. This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

from llmcompressor.modifiers.transform.smoothquant.base import *  # noqa: F401,F403
