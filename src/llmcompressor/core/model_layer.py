from dataclasses import dataclass
from typing import Any

__all__ = ["ModelParameterizedLayer"]


@dataclass
class ModelParameterizedLayer:
    """
    A dataclass for holding a parameter and its layer

    :param layer_name: the name of the layer
    :param layer: the layer object
    :param param_name: the name of the parameter
    :param param: the parameter object
    """

    layer_name: str
    layer: Any
    param_name: str
    param: Any
