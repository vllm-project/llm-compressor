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

__all__ = ["update_fused_layer_weight_global_scales"]


def update_fused_layer_weight_global_scales(submodule: Module):
    """
    When running NVFP4 quantization, update the global scale
    such that q,k,v layers are treated as one tensor with the same
    global_scale and gate_proj/up_proj layers are treated as one tensor
    with the same global scale. This is requirement currently being set
    by vLLM and may be removed in the future OR potentially make it
    an optional step.

    :param model: model to quantize
    """

    # If any of the following sets of layers are found on submodule,
    # and their weights are all TENSOR_GROUP quantized,
    # ensure the global scale is fused
    layers_to_fuse_list = [
        # mlp / expert layers have fused gate_up_proj
        ("gate_proj", "up_proj"),
        # attention layers have fused qkv_proj
        ("q_proj", "k_proj", "v_proj"),
        # DeepSeek multi-latent attention has fused_qkv_a_proj
        ("q_a_proj", "kv_a_proj_with_mqa"),
    ]

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

    for layers_to_fuse in layers_to_fuse_list:
        has_all_layers = all(hasattr(submodule, layer) for layer in layers_to_fuse)
        if not has_all_layers:
            continue

        layers = [getattr(submodule, layer) for layer in layers_to_fuse]
        if not _is_valid_tensor_group_quant(layers):
            continue

        with align_modules(layers):
            global_scale = torch.min(
                torch.cat([layer.weight_global_scale.data for layer in layers])
            ).reshape([1])

            for layer in layers:
                update_offload_parameter(layer, "weight_global_scale", global_scale)

            del global_scale
