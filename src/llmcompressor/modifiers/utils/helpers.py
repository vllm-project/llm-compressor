from typing import List

import torch
from compressed_tensors.quantization import QuantizationStrategy
from compressed_tensors.utils import align_modules, update_parameter_data
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

    def _is_attention_module(module: Module):
        return "attention" in module.__class__.__name__.lower() and (
            hasattr(module, "k_proj")
            or hasattr(module, "v_proj")
            or hasattr(module, "qkv_proj")
        )

    def _is_mlp_module(module: Module):
        return "mlp" in module.__class__.__name__.lower() and (
            hasattr(module, "gate_proj") or hasattr(module, "up_proj")
        )

    def _valid_tensor_group_quant(layer_list: List[Linear]):
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

    if _is_attention_module(submodule):
        # already fused/treated as one layer
        if hasattr(submodule, "qkv_proj"):
            return

        if not _valid_tensor_group_quant(
            [submodule.q_proj, submodule.v_proj, submodule.k_proj]
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

        update_parameter_data(submodule.k_proj, global_scale, "weight_global_scale")
        update_parameter_data(submodule.q_proj, global_scale, "weight_global_scale")
        update_parameter_data(submodule.v_proj, global_scale, "weight_global_scale")

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

        update_parameter_data(submodule.gate_proj, global_scale, "weight_global_scale")
        update_parameter_data(submodule.up_proj, global_scale, "weight_global_scale")

        del global_scale
