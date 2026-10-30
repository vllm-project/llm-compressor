"""
Utilities for REAP: expert pruning.

MoE layer detection and structural expert pruning are shared with the other
expert pruning modifiers and live in
``llmcompressor.modifiers.pruning.utils``; they are re-exported here for
backwards compatibility.
"""

import torch

from llmcompressor.modifiers.pruning.utils import (
    NUM_EXPERTS_CONFIG_KEYS,
    MoeModelAttrs,
    get_moe_attrs,
    prune_moe_layer,
)

__all__ = [
    "REAP_PRECISION",
    "MoeModelAttrs",
    "NUM_EXPERTS_CONFIG_KEYS",
    "get_moe_attrs",
    "prune_moe_layer",
]


REAP_PRECISION = torch.float32
