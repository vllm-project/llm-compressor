"""
Utility / helper functions
"""

import difflib
import re
from operator import attrgetter
from typing import Dict, List, Optional, Tuple, Union

import torch
from compressed_tensors import InternalModule
from compressed_tensors.quantization.utils import is_module_quantized
from torch.nn import Linear, Module, Parameter
from torch.nn.modules.conv import _ConvNd
from transformers import PreTrainedModel

from llmcompressor.core import ModelParameterizedLayer
from llmcompressor.utils.fsdp.context import (
    fix_fsdp_module_name,
    summon_full_params_context,
)
from compressed_tensors import match_named_modules

try:
    quant_err = None
    from torch.nn.qat import Conv2d as QATConv2d
    from torch.nn.qat import Linear as QATLinear
    from torch.quantization import QuantWrapper
except Exception as _err:
    quant_err = _err
    QuantWrapper = None
    QATLinear = None
    QATConv2d = None

try:
    from torch.nn.qat import Conv3d as QATConv3d
except Exception as _err:
    quant_conv3d_err = _err
    QATConv3d = None


try:
    from transformers.modeling_utils import Conv1D as TransformerConv1D
except Exception as _err:
    gpt_conv1d_err = _err
    TransformerConv1D = None


__all__ = [
    "qat_active",
    "get_matching_layer",
    "get_no_split_params",
]

ALL_TARGET = "__ALL__"
ALL_PRUNABLE_TARGET = "__ALL_PRUNABLE__"
ALL_QUANTIZABLE_TARGET = "__ALL_QUANTIZABLE__"


def match_class(layer: Module, targets: Union[str, List[str]]) -> Tuple[bool, int]:
    if isinstance(targets, str):
        targets = [targets]

    for index, target in enumerate(targets):
        if layer.__class__.__name__ == target:
            return True, index

    return False, -1


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


def get_matching_layer(
    target: str, name_to_match: str, module: Module
) -> Optional[Tuple[str, Module]]:
    """
    Given a target regex, find the layer name in the module that most closely matches
    the name_to_match string. This is used to matches submodules in the same layer, for
    instance matching "re.*k_proj" to "model.decoder.layer.0.q_proj" to find the k_proj
    that exists in layer 0.

    :param target: regex to search for
    :param name_to_match: full layer name to match to, should exist in module
    :param module: module to search for target in
    :return: Tuple containing the layer name and module that fits the target regex and
    best matches name_to_match, or None if no match can be found
    """
    potential_matches = match_named_modules(target, module)
    largest_substring = 0
    match = None
    for name, module in potential_matches.items():
        seq_matcher = difflib.SequenceMatcher(None, name, name_to_match)
        _, _, match_length = seq_matcher.find_longest_match(
            0, len(name), 0, len(name_to_match)
        )
        if match_length > largest_substring:
            match = (name, module)
            largest_substring = match_length

    return match


def get_no_split_params(model: PreTrainedModel) -> Union[str, List[str]]:
    """
    Get list of module classes that shouldn't be split when sharding. For
    Hugging Face Transformer models, this is the decoder layer type. For other
    types of models, this just returns all module names.

    :return: list of class names that shouldn't be split
    """
    # importing here to avoid circular import
    from llmcompressor.utils.fsdp.helpers import maybe_get_wrapped

    model = maybe_get_wrapped(model)
    no_split_modules = model._get_no_split_modules("auto")
    if len(no_split_modules) <= 0:
        return ALL_TARGET

    return no_split_modules


# https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8
def get_layer_by_name(layer_name: str, module: Module) -> Module:
    """
    Get the layer of a module by name.
    :param layer_name: Name of the layer to find.
    :param module: Module in which to search for layer_name
    :return: Module, the layer with name layer_name
    """
    return attrgetter(layer_name)(module)
