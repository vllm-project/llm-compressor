from typing import List, Optional, Union

import torch
from torch.nn.modules.linear import Identity
from torch.quantization import QuantWrapper

__all__ = ["get_leaf_operations", "is_quantized", "get_precision_information"]


def get_leaf_operations(
    model: torch.nn.Module,
    operations_to_skip: Optional[List[torch.nn.Module]] = None,
    operations_to_unwrap: Optional[List[torch.nn.Module]] = None,
) -> List[torch.nn.Module]:
    """
    Get the leaf operations in the model
    (those that do not have operations as children)

    :param model: the model to get the leaf operations from
    :param operations_to_skip: a list of leaf operations that will be
        omitted when getting the leaf operations. If None passed, by
        default the Identity operation will be skipped
    :param operations_to_unwrap: a list of operations that will be unwrapped
        when getting the leaf operations. Unwrapping means that we directly
        add the module(s) that is/are wrapped by the operation (i.e. operation's
        `module` attribute) to the list
        of leaf operations. If None passed, by default the QuantWrapper
        operation will be unwrapped
    :return: a list of the leaf operations
    """
    if operations_to_skip is None:
        operations_to_skip = [Identity]

    if operations_to_unwrap is None:
        operations_to_unwrap = [QuantWrapper]

    leaf_operations = []
    children = list(model.children())

    if children == []:
        return model
    else:
        for child in children:
            if isinstance(child, tuple(operations_to_unwrap)):
                leaf_operations.append(child.module)
                continue
            try:
                leaf_operations.extend(get_leaf_operations(child))
            except TypeError:
                leaf_operations.append(get_leaf_operations(child))
    leaf_operations = [
        op for op in leaf_operations if not isinstance(op, tuple(operations_to_skip))
    ]
    return leaf_operations


def is_quantized(operation: torch.nn.Module) -> bool:
    """
    Check whether the operation is quantized (contains
    a quantization scheme)
    """
    return hasattr(operation, "quantization_scheme")


def get_precision_information(
    operation: torch.nn.Module,
) -> Union[None, int, "QuantizationScheme"]:  # noqa F821
    """
    Get the information about the precision of the operation.

    1)  If operation is quantized, returns the quantization
        scheme of the operation.
    2)  If operation is not quantized, returns the numer of bits
        of the operation's weights.
    3)  If operation is not quantized and does not have a weights,
        returns None.

    :param operation: the operation to get the quantization scheme from
    :return: the quantization scheme of the operation, the number of bits
        of the operation's weights, or None if the operation is not quantized
        and does not have a weight
    """

    if hasattr(operation, "quantization_scheme"):
        return getattr(operation, "quantization_scheme")
    elif hasattr(operation, "weight"):
        return _get_num_bits(operation.weight.dtype)
    else:
        return None


def _get_num_bits(dtype: torch.dtype) -> int:
    # Get the number of bits of a torch dtype
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return 16
    elif dtype == torch.float32:
        return 32
    elif dtype == torch.float64:
        return 64
    elif dtype == torch.int8:
        return 8
    elif dtype == torch.int16:
        return 16
    elif dtype == torch.int32:
        return 32
    elif dtype == torch.int64:
        return 64
    else:
        raise ValueError("Unknown dtype: {}".format(dtype))
