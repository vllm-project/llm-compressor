"""
Utility / helper functions
"""

import random
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy
import torch
from torch import Tensor
from torch.nn import Linear, Module

try:
    quant_err = None
except Exception as _err:
    quant_err = _err
    QuantWrapper = None
    QATLinear = None
    QATConv2d = None


__all__ = [
    "tensors_to_device",
    "tensors_to_precision",
    "tensors_module_forward",
    "tensor_sparsity",
    "get_linear_layers",
    "get_quantized_layers",
    "set_deterministic_seeds",
]


##############################
#
# pytorch tensor helper functions
#
##############################


def tensors_to_device(
    tensors: Tensor | Iterable[Tensor] | dict[Any, Tensor], device: str
) -> Tensor | Iterable[Tensor] | dict[Any, Tensor]:
    """
    Default function for putting a tensor or collection of tensors to the proper device.
    Returns the tensor references after being placed on the proper device.

    Supported use cases:
        - single tensor
        - Dictionary of single tensors
        - Dictionary of iterable of tensors
        - Dictionary of dictionary of tensors
        - Iterable of single tensors
        - Iterable of iterable of tensors
        - Iterable of dictionary of tensors

    :param tensors: the tensors or collection of tensors to put onto a device
    :param device: the string representing the device to put the tensors on,
        ex: 'cpu', 'cuda', 'cuda:1'
    :return: the tensors or collection of tensors after being placed on the device
    """
    match tensors:
        case Tensor():
            return tensors.to(device)

        case OrderedDict():
            return OrderedDict(
                [
                    (key, tensors_to_device(tens, device))
                    for key, tens in tensors.items()
                ]
            )

        case Mapping():
            return {
                key: tensors_to_device(tens, device) for key, tens in tensors.items()
            }

        case tuple():
            return tuple(tensors_to_device(tens, device) for tens in tensors)

        case Iterable():
            return [tensors_to_device(tens, device) for tens in tensors]

        case _:
            raise ValueError(
                f"unrecognized type for tensors given of {tensors.__class__.__name__}"
            )


def tensors_to_precision(
    tensors: Tensor | Iterable[Tensor] | dict[Any, Tensor], full_precision: bool
) -> Tensor | Iterable[Tensor] | dict[Any, Tensor]:
    """
    :param tensors: the tensors to change the precision of
    :param full_precision: True for full precision (float 32) and
        False for half (float 16)
    :return: the tensors converted to the desired precision
    """
    match tensors:
        case Tensor():
            return tensors.float() if full_precision else tensors.half()

        case OrderedDict():
            return OrderedDict(
                [
                    (key, tensors_to_precision(tens, full_precision))
                    for key, tens in tensors.items()
                ]
            )

        case Mapping():
            return {
                key: tensors_to_precision(tens, full_precision)
                for key, tens in tensors.items()
            }

        case tuple():
            return tuple(tensors_to_precision(tens, full_precision) for tens in tensors)

        case Iterable():
            return [tensors_to_precision(tens, full_precision) for tens in tensors]

        case _:
            raise ValueError(
                f"unrecognized type for tensors given of {tensors.__class__.__name__}"
            )


# used by calibration function, TODO: remove with data pipelines
def tensors_module_forward(
    tensors: Tensor | Iterable[Tensor] | Mapping[Any, Tensor],
    module: Module,
    check_feat_lab_inp: bool = True,
) -> Any:
    """
    Default function for calling into a model with data for a forward execution.
    Returns the model result.
    Note, if an iterable the features to be passed into the model are considered
    to be at index 0 and other indices are for labels.

    Supported use cases: single tensor,
    iterable with first tensor taken as the features to pass into the model

    :param tensors: the data to be passed into the model, if an iterable the features
        to be passed into the model are considered to be at index 0 and other indices
        are for labels
    :param module: the module to pass the data into
    :param check_feat_lab_inp: True to check if the incoming tensors looks like
        it's made up of features and labels ie a tuple or list with 2 items
        (typical output from a data loader) and will call into the model with just
        the first element assuming it's the features False to not check
    :return: the result of calling into the model for a forward pass
    """
    if isinstance(tensors, (tuple, list)) and len(tensors) == 2 and check_feat_lab_inp:
        # assume if this is a list or tuple of 2 items that it is made up of
        # (features, labels) pass the features into a recursive call for the model
        return tensors_module_forward(tensors[0], module, check_feat_lab_inp=False)

    match tensors:
        case Tensor():
            return module(tensors)

        case Mapping():
            return module(**tensors)

        case Iterable():
            return module(*tensors)

        case _:
            raise ValueError(
                f"unrecognized type for data given of {tensors.__class__.__name__}"
            )


def tensor_sparsity(
    tens: Tensor, dim: None | int | list[int] | tuple[int, ...] = None
) -> Tensor:
    """
    :param tens: the tensor to calculate the sparsity for
    :param dim: the dimension(s) to split the calculations over;
        ex, can split over batch, channels, or combos
    :return: the sparsity of the input tens, ie the fraction of numbers that are zero
    """
    if dim is None:
        zeros = (tens.cpu() == 0).sum()
        total = tens.numel()

        return zeros.float() / float(total)

    if isinstance(dim, int):
        dim = [dim]

    if max(dim) >= len(tens.shape):
        raise ValueError(
            "Unsupported dim given of {} in {} for tensor shape {}".format(
                max(dim), dim, tens.shape
            )
        )

    sum_dims = [ind for ind in range(len(tens.shape)) if ind not in dim]
    zeros = (tens == 0).sum(dim=sum_dims) if sum_dims else tens == 0
    total = numpy.prod(
        [tens.shape[ind] for ind in range(len(tens.shape)) if ind not in dim]
    )

    permute_order = sorted(
        ((d, len(dim) - i - 1) for i, d in enumerate(dim)), reverse=True
    )
    permute = [d[1] for d in permute_order]

    if permute != [i for i in range(len(permute))]:
        # need to permute to get desired dimensions at the front
        zeros = zeros.permute(*permute).contiguous()

    return zeros.float() / float(total)


##############################
#
# pytorch module helper functions
#
##############################


def get_linear_layers(module: Module) -> dict[str, Module]:
    """
    :param module: the module to grab all linear layers for
    :return: a list of all linear layers in the module
    """
    return {
        name: mod for name, mod in module.named_modules() if isinstance(mod, Linear)
    }


def get_quantized_layers(module: Module) -> list[tuple[str, Module]]:
    """
    :param module: the module to get the quantized layers from
    :return: a list containing the names and modules of the quantized layers
        (Embedding, Linear, Conv2d, Conv3d)
    """

    quantized_layers = []
    for name, mod in module.named_modules():
        if hasattr(mod, "quantization_scheme"):
            weight_scheme = getattr(mod.quantization_scheme, "weights", None)
            if weight_scheme is not None and hasattr(mod, "weight"):
                quantized_layers.append((name, mod))

    return quantized_layers


def set_deterministic_seeds(seed: int = 0):
    """
    Manually seeds the numpy, random, and torch packages.
    Also sets torch.backends.cudnn.deterministic to True
    :param seed: the manual seed to use. Default is 0
    """
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
