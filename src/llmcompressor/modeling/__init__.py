# ruff: noqa

"""
Model preparation and fusion utilities for compression workflows.

Provides tools for preparing models for compression including
layer fusion, module preparation, and model structure optimization.
Handles pre-compression transformations and architectural modifications
needed for efficient compression.
"""

# trigger registration
from .deepseek_v3 import CalibrationDeepseekV3MoE  # noqa: F401
from .llama4 import SequentialLlama4TextMoe  # noqa: F401
from .qwen3_moe import CalibrationQwen3MoeSparseMoeBlock  # noqa: F401
from .qwen3_vl_moe import CalibrateQwen3VLMoeTextSparseMoeBlock  # noqa: F401
from .qwen3_next_moe import CalibrationQwen3NextSparseMoeBlock  # noqa: F401
# TODO: add granite4, Qwen3Next

from .fuse import *
from .prepare import *
