"""
Helper functions for modifier operations and weight management.

Provides utility functions for updating layer weights, managing
global scales for quantization, and handling fused layer operations in
neural network compression workflows. Supports specialized quantization
strategies like NVFP4.
"""

import logging

import torch
from compressed_tensors.offload import align_modules, update_offload_parameter
from compressed_tensors.quantization import QuantizationStrategy, is_attention_module
from torch.nn import Module

__all__ = ["update_fused_layer_weight_global_scales"]

logger = logging.getLogger(__name__)

# Fused attention projection groups.
# Each entry is a list of attribute names that should share the same global scale.
# The first matching group is used; order matters.
_ATTENTION_FUSED_GROUPS: list[list[str]] = [
    # Already-fused QKV (e.g. GPT-NeoX, Falcon)
    ["qkv_proj"],
    # Standard separate Q/K/V projections (Llama, Mistral, Qwen, etc.)
    ["q_proj", "k_proj", "v_proj"],
    # DeepSeek V2/V3 MLA: compressed Q + compressed KV
    ["q_a_proj", "kv_a_proj_with_mqa"],
    # DeepSeek V2/V3 MLA: decompressed Q + decompressed KV
    ["q_b_proj", "kv_b_proj"],
]


def _fuse_global_scales(layers: list[Module]):
    """
    Given a list of Linear-like modules, set all of their
    ``weight_global_scale`` parameters to the element-wise minimum.
    """
    with align_modules(layers):
        global_scale = torch.min(
            torch.cat([layer.weight_global_scale.data for layer in layers])
        ).reshape([1])

    for layer in layers:
        update_offload_parameter(layer, "weight_global_scale", global_scale)

    del global_scale


def _valid_tensor_group_quant(layer_list: list[Module]) -> bool:
    """
    Return True if all the modules in *layer_list* are
    TENSOR_GROUP quantized (i.e. they carry an NVFP4-style global scale).
    """
    for layer in layer_list:
        scheme = getattr(layer, "quantization_scheme", None)
        if scheme is None:
            return False

        weight_quant_args = scheme.weights
        if weight_quant_args is None:
            return False

        if weight_quant_args.strategy != QuantizationStrategy.TENSOR_GROUP:
            return False
    return True


def update_fused_layer_weight_global_scales(submodule: torch.nn.Module):
    """
    When running NVFP4 quantization, update the global scale
    such that fused projection layers are treated as one tensor with the same
    global_scale. Specifically:

    * **Attention**: q/k/v projections (or MLA-style compressed projections
      like ``q_a_proj``/``kv_a_proj_with_mqa``) share one global scale.
    * **MLP**: gate_proj and up_proj share one global scale.

    This is a requirement currently being set by vLLM and may be removed in
    the future OR potentially made an optional step.

    :param submodule: a single sub-module of the model (attention or MLP block)
    """

    def _is_mlp_module(module: Module) -> bool:
        return "mlp" in module.__class__.__name__.lower() and (
            hasattr(module, "gate_proj") and hasattr(module, "up_proj")
        )

    # --- Attention fused groups ---
    if is_attention_module(submodule):
        for group in _ATTENTION_FUSED_GROUPS:
            layers = [
                getattr(submodule, name)
                for name in group
                if hasattr(submodule, name)
            ]
            # Only fuse when ALL names in the group are present
            if len(layers) != len(group):
                continue

            # Skip single-projection groups (already fused, e.g. qkv_proj)
            if len(layers) <= 1:
                return

            if not _valid_tensor_group_quant(layers):
                return

            _fuse_global_scales(layers)
            return  # only the first matching group applies

    # --- MLP fused groups ---
    if _is_mlp_module(submodule):
        gate_proj = getattr(submodule, "gate_proj")
        up_proj = getattr(submodule, "up_proj")

        if not _valid_tensor_group_quant([gate_proj, up_proj]):
            return

        _fuse_global_scales([gate_proj, up_proj])
