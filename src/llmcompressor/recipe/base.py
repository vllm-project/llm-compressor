from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from llmcompressor.recipe.args import RecipeArgs

__all__ = ["RecipeBase"]


class RecipeBase(BaseModel, ABC):
    """
    Defines the contract that `Recipe` and its components
    such as `RecipeModifier` and `RecipeStage` must follow.

    All inheritors of this class must implement the following methods:
        - calculate_start
        - calculate_end
        - evaluate
        - create_modifier
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def create_modifier(self) -> Any:
        raise NotImplementedError()
