"""
Utility / helper functions
"""

import torch
from compressed_tensors.quantization.utils import is_module_quantized
from loguru import logger
from torch.nn import Module
from transformers import PreTrainedModel

__all__ = [
    "qat_active",
    "get_no_split_params",
    "infer_sequential_targets",
]

ALL_TARGET = "__ALL__"


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


def get_no_split_params(model: PreTrainedModel) -> str | list[str]:
    """
    Get list of module classes that shouldn't be split when sharding. For
    Hugging Face Transformer models, this is the decoder layer type. For other
    types of models, this just returns all module names.

    :return: list of class names that shouldn't be split
    """
    try:
        # Transformers < v5 support
        no_split_modules = model._get_no_split_modules("auto")
    except AttributeError:
        # Transformers v5 support
        no_split_modules = model._no_split_modules
    if len(no_split_modules) <= 0:
        return ALL_TARGET

    return no_split_modules


def infer_sequential_targets(
    model: Module, sequential_targets: str | list[str] | None = None
) -> str | list[str]:
    """
    Infer or validate sequential targets for layer-wise processing.

    When sequential_targets is None, automatically infers targets using
    get_no_split_params(). When provided as a string, wraps it in a list.
    Otherwise, returns the provided list as-is.

    :param model: The model to infer targets from
    :param sequential_targets: Optional sequential targets to use. If None,
        targets are inferred from the model. If a string, it's wrapped in a list.
    :return: List of sequential target class names or patterns
    """
    match sequential_targets:
        case None:
            return get_no_split_params(model)
        case str():
            return [sequential_targets]
        case _:
            return sequential_targets


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
