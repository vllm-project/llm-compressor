# ruff: noqa

"""
Model preparation and fusion utilities for compression workflows.

Provides tools for preparing models for compression including
layer fusion, module preparation, and model structure optimization.
Handles pre-compression transformations and architectural modifications
needed for efficient compression.
"""

# trigger registration
from .offset_norm import (
    CalibrationOffsetNorm,
    NormCalibrationTransformations,
)

from .fuse import *
from .moe.conversion_mappings import patch_moe_mappings  # noqa: F401
from .moe.linearize import repack_moe  # noqa: F401


def create_model_transformations(model):
    """Prepare registered temporary transformations for a calibration model."""
    return NormCalibrationTransformations(model)
