"""
Helper functions for modifier operations and weight management.

Provides utility functions for updating layer weights, managing
global scales for quantization, and handling fused layer operations in
neural network compression workflows. Supports specialized quantization
strategies like NVFP4.
"""

import torch
from compressed_tensors.quantization import QuantizationStrategy
from compressed_tensors.utils import (
    align_modules,
    update_offload_parameter,
    update_parameter_data,
)
from torch.nn import Linear

from llmcompressor.modeling.fused_modules import (
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
    definitions; see :mod:`llmcompressor.modeling.fused_modules`.

    When a linear already has ``weight_scale`` (e.g. after parallel phase-1
    calibration), per-tensor scale is rescaled so that q = x/(s'*g') is
    unchanged: s' = s * (g' / g), where g' is the fused global scale and g
    was the previous per-tensor global scale.

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
            _apply_fused_global_scale(lin, global_scale)
        del global_scale

    # Fused MLP: gate_proj, up_proj
    linears = get_fused_mlp_linears(submodule)
    if linears is not None and _valid_tensor_group_quant(linears):
        with align_modules(linears):
            global_scale = torch.min(
                torch.cat([lin.weight_global_scale.data for lin in linears])
            ).reshape([1])
        for lin in linears:
            _apply_fused_global_scale(lin, global_scale)
        del global_scale


def _apply_fused_global_scale(lin: Linear, g_prime: torch.Tensor) -> None:
    """Set weight_global_scale to g'; rescale weight_scale so q = x/(s*g) unchanged."""
    old_g = lin.weight_global_scale.data
    update_parameter_data(lin, g_prime, "weight_global_scale")
    weight_scale = getattr(lin, "weight_scale", None)
    if weight_scale is not None:
        # s' = s * (g' / g) so that x / s' / g' = x / s / g
        ratio = (g_prime / old_g).to(weight_scale.dtype).to(weight_scale.device)
        new_scale = weight_scale.data * ratio
        update_offload_parameter(lin, "weight_scale", new_scale)
