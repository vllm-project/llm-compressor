from enum import Enum
from typing import Any, Optional

__all__ = ["UpdateMethod", "validate_sequential_update"]


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
