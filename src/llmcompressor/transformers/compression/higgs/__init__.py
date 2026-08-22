"""
HIGGS: Heuristic ILP-Guided Grouped Scheme Mixed-Precision Quantization
"""

from llmcompressor.transformers.compression.higgs.base import (
    HiggsMSECollectorConverter,
    HiggsQuantizationConverter,
    get_higgs_config,
)
from llmcompressor.transformers.compression.higgs.ilp_solver import (
    solve_ilp_mixed_precision,
)
from llmcompressor.transformers.compression.higgs.utils import (
    compute_heuristic_alphas,
    compute_layer_mse,
    detect_fused_groups,
    generate_config_groups,
)


__all__ = [
    "get_higgs_config",
    "HiggsMSECollectorConverter",
    "HiggsQuantizationConverter",
    "compute_layer_mse",
    "solve_ilp_mixed_precision",
    "generate_config_groups",
    "compute_heuristic_alphas",
    "detect_fused_groups",
]
