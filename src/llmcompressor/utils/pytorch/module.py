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
    "match_targets",
    "get_default_params",
    "match_layers_params",
    "get_layers",
    "get_layer",
    "set_layer",
    "get_params",
    "get_param",
    "get_terminal_layers",
    "get_prunable_layers",
    "get_quantizable_layers",
    "qat_active",
    "get_layers_params",
    "get_matching_layer",
    "get_no_split_params",
    "get_layer_by_name",
]

ALL_TARGET = "__ALL__"
ALL_PRUNABLE_TARGET = "__ALL_PRUNABLE__"
ALL_QUANTIZABLE_TARGET = "__ALL_QUANTIZABLE__"


def match_targets(name: str, targets: Union[str, List[str]]) -> Tuple[bool, int]:
    if isinstance(targets, str):
        targets = [targets]

    for index, target in enumerate(targets):
        if target[:3] == "re:":
            pattern = target[3:]
            if re.match(pattern, name):
                return True, index
        elif name == target:
            return True, index

    return False, -1


def match_class(layer: Module, targets: Union[str, List[str]]) -> Tuple[bool, int]:
    if isinstance(targets, str):
        targets = [targets]

    for index, target in enumerate(targets):
        if layer.__class__.__name__ == target:
            return True, index

    return False, -1


def get_default_params(layers: Dict[str, Module]) -> Dict[str, Parameter]:
    params = {}
    for name, layer in layers.items():
        for param_name, param in layer.named_parameters():
            if param_name == "weight":
                params[name] = param
                break
    return params


def match_layers_params(
    targets: Union[str, List[str]], module: Module, params: bool = False
) -> Dict[str, Union[Module, Parameter]]:
    if targets == ALL_TARGET:
        values = get_terminal_layers(module)

        return values if not params else get_default_params(values)

    if targets == ALL_PRUNABLE_TARGET:
        values = get_prunable_layers(module)

        return values if not params else get_default_params(values)

    if targets == ALL_QUANTIZABLE_TARGET:
        values = get_quantizable_layers(module)

        return values if not params else get_default_params(values)

    if isinstance(targets, str):
        targets = [targets]

    resolved = {}
    targets_found = [False for _ in range(len(targets))]

    for name, layer in module.named_modules():
        # due to nesting, FSDP may not be the top layer
        name = fix_fsdp_module_name(name)
        match, match_index = match_targets(name, targets)
        if match and not params:
            targets_found[match_index] = True
            resolved[name] = layer
        else:
            match, match_index = match_class(layer, targets)
            if match:
                targets_found[match_index] = True
                resolved[name] = layer

        for param_name, param in layer.named_parameters():
            if "." in param_name:  # skip parameters of nested layers
                continue

            param_match, param_match_index = match_targets(
                f"{name}.{param_name}", targets
            )
            if param_match:
                targets_found[param_match_index] = True
                resolved[f"{name}"] = layer if not params else param

    missed = [target for found, target in zip(targets_found, targets) if not found]
    if len(missed) > 0:
        raise ValueError(f"Could not find targets {missed} in module {module}")

    return resolved


def get_layers(
    targets: Union[str, List[str]],
    module: Module,
    exclude_internal_modules: bool = False,
) -> Dict[str, Module]:
    """
    Get layers (also known as submodules) of module based on targets

    :param targets: names or regexes to search for
        Can be regex, e.g. "re:.*input_layernorm$" to find all layers
        in module whose names end in string "input_layernorm"
    :param module: Parent module in which to search for targets
    :param exclude_internal_modules: If True, don't include internal
        modules added by llm-compressor, e.g. Observers and Transforms.
        Defaults to False to maintain backward compatibility

    :return: dict of {layer name -> module} of all layers in module
        that match targets
    """
    layer_dict = match_layers_params(targets, module)
    if exclude_internal_modules:
        layer_dict = {
            name: layer
            for name, layer in layer_dict.items()
            if not isinstance(layer, InternalModule)
        }

    return layer_dict


def get_layer(target: str, module: Module) -> Tuple[str, Module]:
    layers = get_layers(target, module)
    if len(layers) != 1:
        raise ValueError(f"Expected 1 layer for target {target}, found {len(layers)}")
    name, layer = next(iter(layers.items()))

    return name, layer


def set_layer(target: str, layer: Module, module: Module) -> Module:
    with summon_full_params_context(module):
        # importing here to avoid circular import
        from llmcompressor.utils.fsdp.helpers import maybe_get_wrapped

        parent_target = ".".join(target.split(".")[:-1])
        if parent_target != "":
            parent_layer = get_layer(parent_target, module)[1]
        else:
            parent_layer = maybe_get_wrapped(module)
        old_layer = getattr(parent_layer, target.split(".")[-1])
        setattr(parent_layer, target.split(".")[-1], layer)

    return old_layer


def get_params(targets: Union[str, List[str]], module: Module) -> Dict[str, Parameter]:
    return match_layers_params(targets, module, params=True)


def get_param(target: str, module: Module) -> Tuple[str, Parameter]:
    params = get_params(target, module)
    if len(params) != 1:
        raise ValueError(
            f"Expected 1 parameter for target {target}, found {len(params)}"
        )
    name, param = next(iter(params.items()))

    return name, param


def get_terminal_layers(module: Module) -> Dict[str, Module]:
    terminal = {}

    for name, layer in module.named_modules():
        if len(list(layer.named_modules())) > 1:
            continue

        terminal[name] = layer

    return terminal


def get_prunable_layers(module: Module) -> Dict[str, Module]:
    prunable = {}

    for name, layer in module.named_modules():
        if (
            isinstance(layer, Linear)
            or isinstance(layer, _ConvNd)
            or (QATLinear and isinstance(layer, QATLinear))
            or (QATConv2d and isinstance(layer, QATConv2d))
            or (QATConv3d and isinstance(layer, QATConv3d))
            or (TransformerConv1D and isinstance(layer, TransformerConv1D))
        ):
            prunable[name] = layer

    return prunable


def get_quantizable_layers(module: Module) -> Dict[str, Module]:
    if QATLinear is None:
        raise ImportError(
            "PyTorch version is not setup for Quantization. "
            "Please install a QAT compatible version of PyTorch"
        )

    quantizable = {}

    for name, layer in module.named_modules():
        if isinstance(layer, Linear) or isinstance(layer, _ConvNd):
            quantizable[name] = layer

    return quantizable


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


def get_layers_params(
    targets: Union[str, List[str]], module: Module
) -> Dict[str, ModelParameterizedLayer]:
    params = get_params(targets, module)
    layers = get_layers(targets, module)

    parameterized_layers = {}
    for name, param in params.items():
        param_layer = ModelParameterizedLayer(
            layer_name=name, layer=layers[name], param_name=name, param=param
        )
        parameterized_layers[name] = param_layer

    return parameterized_layers


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
    potential_matches = get_layers(target, module)
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
