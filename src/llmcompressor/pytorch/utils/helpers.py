"""
Utility / helper functions
"""

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

from torch import Tensor
from torch.nn import Module

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
