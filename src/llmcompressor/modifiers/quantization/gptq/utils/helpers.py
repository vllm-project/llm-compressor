from enum import Enum
from typing import Any, Iterable, Optional

import torch

__all__ = ["UpdateMethod", "validate_sequential_update", "get_output_error"]


class UpdateMethod(str, Enum):
    """
    Enum storing different methods by which to sequentially update the weights
    of a model


    Module: apply compression and proceed with weight-quantized outputs for each
        module within a transformer block (layer). NOT IMPLEMENTED\n
    Layer: apply compression and proceed with weight-quantized outputs for each
        transformer block (layer)\n
    """

    MODULE = "module"
    LAYER = "layer"


def validate_sequential_update(value: Any) -> Optional[UpdateMethod]:
    if isinstance(value, bool):
        return UpdateMethod.LAYER if value else None

    if isinstance(value, str):
        value = UpdateMethod(value.lower())

    if value == UpdateMethod.MODULE:
        raise ValueError(
            'sequential_update="module" is not supported in the current version'
        )

    return value


def get_output_error(unquantized, quantized):
    unquantized_outputs = sum(
        [
            [output for output in outputs]
            if isinstance(outputs, Iterable)
            else [outputs]
            for outputs, _ in unquantized
        ],
        start=[],
    )

    quantized_outputs = sum(
        [
            [output for output in outputs]
            if isinstance(outputs, Iterable)
            else [outputs]
            for outputs, _ in quantized
        ],
        start=[],
    )

    assert len(unquantized_outputs) == len(quantized_outputs)
    return sum(
        [
            torch.nn.functional.l1_loss(unq, q)
            for unq, q in zip(unquantized_outputs, quantized_outputs)
        ]
    )
