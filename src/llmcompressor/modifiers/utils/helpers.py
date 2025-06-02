from compressed_tensors.quantization.utils import (
    is_fp4,
    iter_named_quantizable_modules,
)
import torch
from torch.nn import Module, Linear
from compressed_tensors.utils import update_parameter_data
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam

def update_fused_layer_weight_global_scales(model: torch.nn.Module):
    """
    When running NVFP4A16 quantization, update the global scale
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

    def _valid_fp4_quant(layer_list: List[Linear]):
        """
        Return True if all the linear layers in the layer_list are
        NVFP4A16 quantized.
        """
        for layer in layer_list:
            scheme = getattr(layer, "quantization_scheme", None)
            if scheme is None:
                return False

            weight_quant_args = scheme.weights

            if weight_quant_args is None:
                return False

            if not is_fp4(quantization_args=weight_quant_args):
                return False
        return True

    for name, submodule in iter_named_quantizable_modules(
        model,
        include_attn=True,
        include_mlp=True,
    ):

        if _is_attention_module(submodule):
            # already fused/treated as one layer
            if hasattr(submodule, "qkv_proj"):
                continue

            if not _valid_fp4_quant(
                [submodule.q_proj, submodule.v_proj, submodule.k_proj]
            ):
                continue

            q_weight = submodule.q_proj.weight.data
            v_weight = submodule.v_proj.weight.data
            k_weight = submodule.k_proj.weight.data

            value = generate_global_scale(
                input_tensor=torch.cat((q_weight, v_weight, k_weight), dim=0)
            )

            update_parameter_data(submodule.q_proj, value, "weight_global_scale")
            update_parameter_data(submodule.k_proj, value, "weight_global_scale")
            update_parameter_data(submodule.v_proj, value, "weight_global_scale")

        if _is_mlp_module(submodule):
            if not _valid_fp4_quant([submodule.gate_proj, submodule.up_proj]):
                continue

            gate_data = submodule.gate_proj.weight.data
            up_data = submodule.up_proj.weight.data

            value = generate_global_scale(
                input_tensor=torch.cat((gate_data, up_data), dim=0)
            )

            update_parameter_data(submodule.gate_proj, value, "weight_global_scale")
            update_parameter_data(submodule.up_proj, value, "weight_global_scale")