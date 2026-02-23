"""
Helper functions for modifier operations and weight management.

Provides utility functions for updating layer weights, managing
global scales for quantization, and handling fused layer operations in
neural network compression workflows. Supports specialized quantization
strategies like NVFP4.
"""

import torch
from compressed_tensors.quantization import QuantizationStrategy
from compressed_tensors.utils import align_modules, update_parameter_data
from torch.nn import Linear

from llmcompressor.modifiers.utils.fused_modules import (
    get_fused_attention_linears,
    get_fused_mlp_linears,
)

__all__ = ["update_fused_layer_weight_global_scales"]


def _valid_tensor_group_quant(layer_list: list[Linear]) -> bool:
    """Return True if all linear layers in layer_list are TENSOR_GROUP quantized."""
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
    When running NVFP4 quantization, update the global scale so that vLLM
    fused groups share one global scale: attention (traditional q/k/v or
    MLA q_a + kv_a) and MLP (gate/up). Uses the centralized fused module
    definitions; see :mod:`llmcompressor.modifiers.utils.fused_modules`.

    This is a requirement currently set by vLLM and may be removed or
    made optional in the future.

    :param submodule: Module to process (any module; only fused attention/MLP
        containers are updated).
    """
    # Fused attention: traditional (q/k/v) or MLA (q_a + kv_a_proj_with_mqa)
    linears = get_fused_attention_linears(submodule)
    if linears is not None and _valid_tensor_group_quant(linears):
        with align_modules(linears):
            global_scale = torch.min(
                torch.cat([lin.weight_global_scale.data for lin in linears])
            ).reshape([1])
        for lin in linears:
            update_parameter_data(lin, global_scale, "weight_global_scale")
        del global_scale

    # Fused MLP: gate_proj, up_proj
    linears = get_fused_mlp_linears(submodule)
    if linears is not None and _valid_tensor_group_quant(linears):
        with align_modules(linears):
            global_scale = torch.min(
                torch.cat([lin.weight_global_scale.data for lin in linears])
            ).reshape([1])
        for lin in linears:
            update_parameter_data(lin, global_scale, "weight_global_scale")
        del global_scale
