from typing import Any, Dict, Optional

from pydantic import model_validator

from llmcompressor.modifiers import Modifier, ModifierFactory
from llmcompressor.recipe.base import RecipeBase

__all__ = ["RecipeModifier"]


class RecipeModifier(RecipeBase):
    """
    A RecipeModifier is a modifier that is defined in a recipe and can be
    evaluated and used to create a  Modifier instance using
    the ModifierFactory.

    :param type: the type of modifier to create
    :param group: the group to assign the modifier to
    :param args: the args to use for the modifier
    """

    type: str
    group: Optional[str] = None
    args: Optional[Dict[str, Any]] = None

    def create_modifier(self) -> "Modifier":
        """
        Create a Modifier instance using the ModifierFactory

        :return: the created modifier
        """
        if not ModifierFactory._loaded:
            ModifierFactory.refresh()
        return ModifierFactory.create(
            self.type,
            allow_registered=True,
            allow_experimental=True,
            **self.args,
        )

    @model_validator(mode="before")
    @classmethod
    def extract_modifier_type(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if len(values) == 2:
            if "group" not in values:
                raise ValueError(
                    "Invalid format: expected keys 'group' and one modifier "
                    f"type, but got keys: {list(values.keys())}"
                )

            # values contains only group and the Modifier type as keys
            group = values.pop("group")
            modifier_type, args = values.popitem()
            return {"group": group, "type": modifier_type, "args": args}

        # values already in the correct format
        return values

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        :return: the dictionary representation of the modifier
        """
        return {self.type: self.args, "group": f"{self.group}_modifiers"}
