# ruff: noqa

"""
Model preparation and fusion utilities for compression workflows.

Provides tools for preparing models for compression including
layer fusion, module preparation, and model structure optimization.
Handles pre-compression transformations and architectural modifications
needed for efficient compression.
"""

# trigger registration
from .offset_norm import CalibrationOffsetNorm  # noqa: F401

from .fuse import *
from .moe.conversion_mappings import patch_moe_mappings  # noqa: F401
