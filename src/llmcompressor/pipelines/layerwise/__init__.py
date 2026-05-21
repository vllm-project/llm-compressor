# ruff: noqa

"""
Layerwise calibration pipeline for memory-efficient quantization.

Enables quantization of models that are too large to fit in memory by loading
weights per-subgraph from safetensors files during calibration.
"""

from .pipeline import *
