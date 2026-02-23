"""
Helper functions for modifier operations and weight management.

Provides utility functions for updating layer weights, managing
global scales for quantization, and handling fused layer operations in
neural network compression workflows. Supports specialized quantization
strategies like NVFP4.
"""

import torch
from compressed_tensors.offload import align_modules, update_offload_parameter
from compressed_tensors.quantization import QuantizationStrategy, is_attention_module
from torch.nn import Linear, Module

__all__ = ["update_fused_layer_weight_global_scales"]


def update_fused_layer_weight_global_scales(submodule: torch.nn.Module):
    """
    When running NVFP4 quantization, update the global scale
    such that q,k,v layers are treated as one tensor with the same
    global_scale and gate_proj/up_proj layers are treated as one tensor
    with the same global scale. This is requirement currently being set
    by vLLM and may be removed in the future OR potentially make it
    an optional step.

    :param model: model to quantize
    """

    def _is_mlp_module(module: Module):
        return "mlp" in module.__class__.__name__.lower() and (
            hasattr(module, "gate_proj") and hasattr(module, "up_proj")
        )

    def _valid_tensor_group_quant(layer_list: list[Linear]):
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

    if is_attention_module(submodule):
        # already fused/treated as one layer
        if hasattr(submodule, "qkv_proj"):
            return

        # not traditional attention (TODO: MLA)
        if not (
            hasattr(submodule, "q_proj")
            and hasattr(submodule, "k_proj")
            and hasattr(submodule, "v_proj")
        ):
            return

        if not _valid_tensor_group_quant(
            [submodule.q_proj, submodule.k_proj, submodule.v_proj]
        ):
            return

        with align_modules([submodule.q_proj, submodule.v_proj, submodule.k_proj]):
            global_scale = torch.min(
                torch.cat(
                    (
                        submodule.q_proj.weight_global_scale.data,
                        submodule.k_proj.weight_global_scale.data,
                        submodule.v_proj.weight_global_scale.data,
                    )
                )
            ).reshape([1])

        update_offload_parameter(submodule.k_proj, "weight_global_scale", global_scale)
        update_offload_parameter(submodule.q_proj, "weight_global_scale", global_scale)
        update_offload_parameter(submodule.v_proj, "weight_global_scale", global_scale)

        del global_scale

    if _is_mlp_module(submodule):
        if not _valid_tensor_group_quant([submodule.gate_proj, submodule.up_proj]):
            return

        with align_modules([submodule.gate_proj, submodule.up_proj]):
            global_scale = torch.min(
                torch.cat(
                    (
                        submodule.gate_proj.weight_global_scale.data,
                        submodule.up_proj.weight_global_scale.data,
                    )
                )
            ).reshape([1])

        update_offload_parameter(
            submodule.gate_proj,
            "weight_global_scale",
            global_scale,
        )
        update_offload_parameter(submodule.up_proj, "weight_global_scale", global_scale)

        del global_scale
