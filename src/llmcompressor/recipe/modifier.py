from typing import Any, Dict, Optional

from pydantic import model_validator

from llmcompressor.modifiers import Modifier, ModifierFactory
from llmcompressor.recipe.args import RecipeArgs
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
    :param args_evaluated: the evaluated args for the modifier
    """

    type: str
    group: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    args_evaluated: Optional[Dict[str, Any]] = None

    def calculate_start(self) -> int:
        """
        :raises: ValueError if args have not been evaluated
        :return: the start epoch for the modifier, -1 if no start is defined
        """
        if not self.args_evaluated:
            raise ValueError("args must be evaluated before calculating start")

        return self.args_evaluated.get("start", -1)

    def calculate_end(self) -> int:
        """
        :raises: ValueError if args have not been evaluated
        :return: the end epoch for the modifier, -1 if no end is defined
        """
        if not self.args_evaluated:
            raise ValueError("args must be evaluated before calculating end")

        return self.args_evaluated.get("end", -1)

    def evaluate(self, args: Optional[RecipeArgs] = None, shift: Optional[int] = None):
        """
        Evaluate the args for the modifier and shift the start and end if provided

        :param args: the args to use for evaluation
        :param shift: the amount to shift the start and end by
        """
        if not self.args:
            raise ValueError("args must be set before evaluating")

        comb_args = args or RecipeArgs()
        self.args_evaluated = comb_args.evaluate_ext(self.args)

        if shift is not None and self.args_evaluated.get("start") is not None:
            self.args_evaluated["start"] += shift

        if shift is not None and self.args_evaluated.get("end") is not None:
            self.args_evaluated["end"] += shift

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
        modifier = {"group": values.pop("group")}
        assert len(values) == 1, "multiple key pairs found for modifier"
        modifier_type, args = list(values.items())[0]

        modifier["type"] = modifier_type
        modifier["args"] = args
        return modifier

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        :return: the dictionary representation of the modifier
        """
        return {self.type: self.args_evaluated, "group": f"{self.group}_modifiers"}
