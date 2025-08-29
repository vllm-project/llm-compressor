"""
Constants for modifier operations and compression thresholds.

This module defines global constants used throughout the compression
framework for determining sparsity thresholds, pruning criteria, and
other modifier-specific parameters.
"""

__all__ = ["SPARSITY_THRESHOLD"]

SPARSITY_THRESHOLD: float = 0.05
