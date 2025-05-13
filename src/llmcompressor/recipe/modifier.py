from typing import Any, Dict, Optional

from pydantic import model_validator

from llmcompressor.modifiers import Modifier, ModifierFactory
from llmcompressor.recipe.base import RecipeBase
from llmcompressor.recipe.utils import evaluate_ext, eval_args 

__all__ = ["RecipeModifier"]


class RecipeModifier(RecipeBase):
    """
    A RecipeModifier is a modifier that is defined in a recipe and can be
    evaluated and used to create a  Modifier instance using
    the ModifierFactory.

    :param type: the type of modifier to create
    :param group: the group to assign the modifier to
    :param args: the args to use for the modifier
    :param args_evaluated: the evaluated args for the modifier
    """

    type: str
    group: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    args_evaluated: Optional[Dict[str, Any]] = None

    def evaluate(self, args: Optional[Dict[str, Any]] = None):
        """
        Evaluate the args for the modifier and shift the start and end if provided

        :param args: the args to use for evaluation
        :param shift: the amount to shift the start and end by
        """
        if not self.args:
            raise ValueError("args must be set before evaluating")

        print("-------- modifier evaluate args --------")
        print(args)
        print(self.args)

        context_args = eval_args(args or {})
        self.args_evaluated = evaluate_ext(self.args, context_args)


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
            **self.args_evaluated,
        )

    @model_validator(mode="before")
    @classmethod
    def extract_modifier_type(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        print("----------- modifier values ---------")
        print(values)
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
        return {self.type: self.args_evaluated, "group": f"{self.group}_modifiers"}
