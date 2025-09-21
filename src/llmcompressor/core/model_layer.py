"""
Model layer utility classes for LLM compression workflows.

Provides dataclass containers for managing model layers and their associated
parameters during compression operations. Facilitates tracking and manipulation
of specific model components and their parameters.
"""

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
