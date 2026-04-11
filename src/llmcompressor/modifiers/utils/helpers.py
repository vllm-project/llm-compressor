"""
Helper functions for modifier operations and weight management.

Provides utility functions for updating layer weights, managing
global scales for quantization, and handling fused layer operations in
neural network compression workflows. Supports specialized quantization
strategies like NVFP4.
"""

import torch
from compressed_tensors.offload import align_modules, update_offload_parameter
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Linear, Module

__all__ = [
    "update_fused_layer_weight_global_scales",
    "fuse_global_scales_and_adjust",
    "FUSED_LAYER_NAMES",
]

# Defines which layer names should have their global_scale fused together.
# These sets are used for TENSOR_GROUP quantization (e.g., NVFP4).
FUSED_LAYER_NAMES = [
    # MLP / expert layers have fused gate_up_proj
    ("gate_proj", "up_proj"),
    # Attention layers have fused qkv_proj
    ("q_proj", "k_proj", "v_proj"),
    # DeepSeek multi-latent attention has fused_qkv_a_proj
    ("q_a_proj", "kv_a_proj_with_mqa"),
    # MoE expert layers may use w1/w3 naming
    ("w1", "w3"),
]


def fuse_global_scales_and_adjust(layers: list[Linear]) -> torch.Tensor:
    """
    Fuse global_scale across multiple layers and adjust weight_scale accordingly.

    For TENSOR_GROUP quantization, this ensures all layers use the same global_scale
    (the minimum across all layers) while preserving the quantization by adjusting
    weight_scale proportionally.

    The relationship is: stored_scale = full_scale * global_scale
    When global_scale changes, weight_scale must be adjusted:
        new_weight_scale = old_weight_scale * (new_global_scale / old_global_scale)

    :param layers: list of Linear layers to fuse (must have weight_global_scale)
    :return: fused global_scale value
    """
    if not layers:
        raise ValueError("Cannot fuse empty list of layers")

    with align_modules(layers):
        # Compute fused global_scale (minimum across all layers)
        global_scales = [layer.weight_global_scale.data for layer in layers]
        fused_global_scale = torch.min(torch.cat(global_scales, dim=0)).reshape([1])

        # Adjust weight_scale and update global_scale for each layer
        for layer in layers:
            old_global_scale = layer.weight_global_scale.data

            # Adjust weight_scale to compensate for change in global_scale
            scale_adjustment = fused_global_scale / old_global_scale
            layer.weight_scale.data.mul_(scale_adjustment)

            # Update global_scale
            update_offload_parameter(layer, "weight_global_scale", fused_global_scale)

    return fused_global_scale


def update_fused_layer_weight_global_scales(submodule: Module):
    """
    When running NVFP4 quantization, update the global scale (and adjust weight scale)
    such that q,k,v layers are treated as one tensor with the same
    global_scale and gate_proj/up_proj layers are treated as one tensor
    with the same global scale. This is requirement currently being set
    by vLLM and may be removed in the future OR potentially make it
    an optional step.

    :param model: model to quantize
    """

    def _is_valid_tensor_group_quant(layer_list: list[Linear]) -> bool:
        """
        Return True if all the linear layers in the layer_list are
        TENSOR_GROUP quantized.
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

    for layers_to_fuse in FUSED_LAYER_NAMES:
        has_all_layers = all(hasattr(submodule, layer) for layer in layers_to_fuse)
        if not has_all_layers:
            continue

        layers = [getattr(submodule, layer) for layer in layers_to_fuse]
        if not _is_valid_tensor_group_quant(layers):
            continue

        # Fuse global scales and adjust weight scales
        fuse_global_scales_and_adjust(layers)
