# ruff: noqa
"""
Backwards compatibility shim for SmoothQuantModifier utils module.

This module has been moved to llmcompressor.modifiers.transform.smoothquant.utils.
This shim will be removed in a future version.
"""

import warnings

warnings.warn(
    "Importing from 'llmcompressor.modifiers.smoothquant.utils' is deprecated. "
    "Please update your imports to use 'llmcompressor.modifiers.transform.smoothquant.utils' "
    "instead. This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

from llmcompressor.modifiers.transform.smoothquant.utils import *  # noqa: F401,F403
