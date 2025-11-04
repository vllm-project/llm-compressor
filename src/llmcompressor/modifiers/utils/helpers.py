"""
Helper functions for modifier operations and weight management.

Provides utility functions for updating layer weights, managing
global scales for quantization, and handling fused layer operations in
neural network compression workflows. Supports specialized quantization
strategies like NVFP4.
"""

from typing import Dict, List, Optional, Tuple

import torch
from compressed_tensors.quantization import QuantizationStrategy
from compressed_tensors.utils import align_modules, update_parameter_data
from torch.nn import Linear, Module

from llmcompressor.utils.helpers import getattr_chain

__all__ = [
    "update_fused_layer_weight_global_scales",
    "validate_group_size_divisibility",
]


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
            hasattr(module, "gate_proj") and hasattr(module, "up_proj")
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


def validate_group_size_divisibility(
    modules: List[Tuple[str, Module]],
    current_ignore_list: List[str],
) -> None:
    """
    Validates that weight columns are evenly divisible by group_size for all
    modules with GROUP strategy quantization. Raises a ValueError if validation
    fails, providing a comprehensive error message with suggested fixes.

    :param modules: List of (name, module) tuples to validate
    :param current_ignore_list: Current list of ignored layer names
    :raises ValueError: If any module has columns not divisible by group_size
    """
    problematic_layers = []

    for module_name, module in modules:
        issue = _check_module_group_size_divisibility(module, module_name)
        if issue is not None:
            problematic_layers.append(issue)

    if problematic_layers:
        error_msg = _generate_validation_error_message(
            problematic_layers, current_ignore_list
        )
        raise ValueError(error_msg)


def _check_module_group_size_divisibility(
    module: Module, module_name: str
) -> Optional[Dict[str, any]]:
    """
    Checks a single module for group size divisibility.

    :param module: Module to check
    :param module_name: Name of the module
    :return: Dict with issue details if not divisible, None otherwise
    """
    quant_args = getattr_chain(module, "quantization_scheme.weights", None)
    if quant_args is None:
        return None

    # Only validate for GROUP strategy
    if quant_args.strategy != QuantizationStrategy.GROUP:
        return None

    # Check if module has weight
    if not hasattr(module, "weight"):
        return None

    weight = module.weight
    group_size = quant_args.group_size

    # Get number of columns based on module type
    num_columns = _get_module_columns(module, weight)
    if num_columns is None:
        return None

    # Check divisibility
    if num_columns % group_size != 0:
        return {
            "name": module_name,
            "num_columns": num_columns,
            "group_size": group_size,
        }

    return None


def _get_module_columns(module: Module, weight: torch.Tensor) -> Optional[int]:
    """
    Get the number of columns for a module based on its type.

    :param module: Module to check
    :param weight: Module's weight tensor
    :return: Number of columns, or None if cannot determine
    """
    if isinstance(module, torch.nn.Linear):
        return weight.shape[1]
    elif isinstance(module, torch.nn.Conv2d):
        return weight.flatten(1).shape[1]
    else:
        # Handle MoE modules with 3D weights [num_experts, out_dim, in_dim]
        # For grouped quantization, we need to check the input dimension (last dim)
        if len(weight.shape) == 3:
            return weight.shape[2]
        elif len(weight.shape) >= 2:
            return weight.shape[1]
        else:
            return None


def _generate_validation_error_message(
    problematic_layers: List[Dict[str, any]], current_ignore_list: List[str]
) -> str:
    """
    Generate comprehensive error message for group size validation failures.

    :param problematic_layers: List of dicts with layer info (name, num_columns, group_size)
    :param current_ignore_list: Current ignore list
    :return: Formatted error message
    """
    error_msg = (
        "\n"
        + "=" * 80
        + "\n"
        "ERROR: Group size divisibility validation failed!\n"
        + "=" * 80
        + "\n\n"
        "The following layers have weight columns that are not evenly divisible\n"
        "by the specified group_size. This will cause failures when running on vLLM.\n\n"
    )

    # Show problematic layers
    error_msg += "Problematic layers:\n"
    for layer in problematic_layers:
        error_msg += (
            f"  - {layer['name']}: {layer['num_columns']} columns "
            f"(not divisible by group_size={layer['group_size']})\n"
        )

    # Build suggested ignore list
    problematic_names = [layer["name"] for layer in problematic_layers]

    error_msg += "\n" + "-" * 80 + "\n"
    error_msg += "Current ignore list:\n"
    if current_ignore_list:
        error_msg += f"  {current_ignore_list}\n"
    else:
        error_msg += "  (empty)\n"

    error_msg += "\nLayers with divisibility issues:\n"
    error_msg += f"  {problematic_names}\n"

    # Combine lists, removing duplicates while preserving order
    combined_ignore = list(current_ignore_list)
    for name in problematic_names:
        if name not in combined_ignore:
            combined_ignore.append(name)

    error_msg += "\n" + "-" * 80 + "\n"
    error_msg += "SUGGESTED FIX: Add the following to your modifier config:\n\n"
    error_msg += f"  ignore: {combined_ignore}\n"
    error_msg += "\n" + "-" * 80 + "\n"
    error_msg += "\nAlternatively, you can disable this validation by setting:\n"
    error_msg += "  validate_group_size: False\n"
    error_msg += "\n(Note: Disabling validation may result in runtime errors on vLLM)\n"
    error_msg += "=" * 80 + "\n"

    return error_msg
