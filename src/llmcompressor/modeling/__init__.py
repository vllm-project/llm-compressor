# ruff: noqa

"""
Model preparation and fusion utilities for compression workflows.

Provides tools for preparing models for compression including
layer fusion, module preparation, and model structure optimization.
Handles pre-compression transformations and architectural modifications
needed for efficient compression.
"""

# trigger registration
from .afmoe import CalibrationAfmoeMoE  # noqa: F401
from .deepseek_v3 import CalibrationDeepseekV3MoE  # noqa: F401
from .deepseek_v4 import CalibrationDeepseekV4MoE  # noqa: F401
from .glm4_moe import CalibrationGlm4MoeMoE  # noqa: F401
from .glm4_moe_lite import CalibrationGlm4MoeLiteMoE  # noqa: F401
from .glm_moe_dsa import CalibrationGlmMoeDsaMoE  # noqa: F401
from .llama4 import SequentialLlama4TextMoe  # noqa: F401
from .qwen3_moe import CalibrationQwen3MoeSparseMoeBlock  # noqa: F401
from .qwen3_5_moe import CalibrationQwen3_5MoeSparseMoeBlock
from .qwen3_vl_moe import CalibrateQwen3VLMoeTextSparseMoeBlock  # noqa: F401
from .qwen3_next_moe import CalibrationQwen3NextSparseMoeBlock  # noqa: F401
from .offset_norm import CalibrationOffsetNorm  # noqa: F401
from .gemma4 import SequentialGemma4TextExperts  # noqa: F401
# TODO: add granite4

from .fuse import *
