"""
Utility / helper functions
"""

import warnings
from typing import Dict, List, Union

import torch
from compressed_tensors.quantization.utils import is_module_quantized
from compressed_tensors.utils import match_named_parameters
from loguru import logger
from torch.nn import Module
from transformers import PreTrainedModel

from llmcompressor.core import ModelParameterizedLayer


__all__ = [
    "expand_special_targets",
    "build_parameterized_layers",
    "qat_active",
    "get_no_split_params",
]

ALL_TARGET = "__ALL__"
ALL_PRUNABLE_TARGET = "__ALL_PRUNABLE__"
ALL_QUANTIZABLE_TARGET = "__ALL_QUANTIZABLE__"


def expand_special_targets(targets: Union[str, List[str]]) -> List[str]:
    """
    Expand special target constants to explicit class names with backward compatibility.

    Special constants like __ALL_PRUNABLE__ and __ALL_QUANTIZABLE__ are deprecated
    in favor of explicit class name lists. This function provides backward compatibility
    by expanding these constants while issuing deprecation warnings.

    :param targets: Target strings which may include special constants
    :return: List of expanded target strings
    :raises ValueError: If __ALL__ constant is used (no longer supported)
    """
    if isinstance(targets, str):
        targets = [targets]

    expanded = []
    for target in targets:
        if target == ALL_PRUNABLE_TARGET:
            warnings.warn(
                f"{ALL_PRUNABLE_TARGET} is deprecated. "
                "Use explicit targets: ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']",
                DeprecationWarning,
                stacklevel=3,
            )
            expanded.extend(["Linear", "Conv1d", "Conv2d", "Conv3d"])
        elif target == ALL_QUANTIZABLE_TARGET:
            warnings.warn(
                f"{ALL_QUANTIZABLE_TARGET} is deprecated. "
                "Use explicit targets: ['Linear', 'Conv2d', 'Conv3d']",
                DeprecationWarning,
                stacklevel=3,
            )
            expanded.extend(["Linear", "Conv2d", "Conv3d"])
        elif target == ALL_TARGET:
            raise ValueError(
                f"{ALL_TARGET} is no longer supported. "
                "Use explicit layer types or patterns instead."
            )
        else:
            expanded.append(target)

    return expanded


def build_parameterized_layers(
    model: Module,
    targets: Union[str, List[str]],
    param_name: str = "weight",
) -> Dict[str, ModelParameterizedLayer]:
    """
    Build ModelParameterizedLayer objects for modules matching the given targets.

    This function replaces get_layers_params() by using compressed-tensors'
    match_named_parameters() to find matching modules and their parameters,
    then constructing ModelParameterizedLayer objects.

    :param model: The model to search for matching modules
    :param targets: Target patterns to match (supports class names, regex with "re:",
                    and special constants for backward compatibility)
    :param param_name: Name of the parameter to extract from each layer (default: "weight")
    :return: Dictionary mapping layer names to ModelParameterizedLayer objects
    """
    # Expand special constants if present
    targets = expand_special_targets(targets)

    parameterized_layers = {}
    for fqn, parent_module, param in match_named_parameters(model, targets):
        # Filter to only the desired parameter (default: "weight")
        if not fqn.endswith(f".{param_name}"):
            continue

        # Extract layer name by removing parameter suffix
        layer_name = fqn.rsplit(".", 1)[0]

        # Avoid duplicate entries (same layer can be matched multiple times)
        if layer_name not in parameterized_layers:
            parameterized_layers[layer_name] = ModelParameterizedLayer(
                layer_name=layer_name,
                layer=parent_module,
                param_name=fqn,
                param=param,
            )

    return parameterized_layers


def qat_active(module: Module) -> bool:
    """
    Determines if any layers in the model have quantization enabled by checking for
    weight_fake_quant attributes

    :param module: PyTorch model to check for quantization
    :return: True if quantization is active anywhere in the model, False otherwise
    """
    for _, layer in module.named_modules():
        if isinstance(layer, torch.quantization.FakeQuantize):
            return True
        if is_module_quantized(layer):
            return True

    return False


def get_no_split_params(model: PreTrainedModel) -> Union[str, List[str]]:
    """
    Get list of module classes that shouldn't be split when sharding. For
    Hugging Face Transformer models, this is the decoder layer type. For other
    types of models, this just returns all module names.

    :return: list of class names that shouldn't be split
    """
    no_split_modules = model._get_no_split_modules("auto")
    if len(no_split_modules) <= 0:
        return ALL_TARGET

    return no_split_modules


# https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8


def get_module_to_name_dict(model: Module) -> dict[Module, str]:
    module_to_name = {}
    for name, module in model.named_modules():
        if module in module_to_name:
            logger.warning(
                f"Warning, {name} and {module_to_name[module]} both "
                "share the same module, which can result in unexpected behavior"
            )
        module_to_name[module] = name
    return module_to_name
