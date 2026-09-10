"""
Evaluation utilities for measuring quantization quality.

Provides tools for assessing the impact of compression on model output
distributions, including KL Divergence-based metrics for comparing quantized
models against their unquantized baselines.
"""

from .kld import KLDivergenceEvaluator, evaluate_kl_divergence

__all__ = [
    "KLDivergenceEvaluator",
    "evaluate_kl_divergence",
]
