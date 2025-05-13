import math
import re
from typing import Any, Dict, Optional, Union

__all__ = ["RecipeArgs"]


class RecipeArgs(Dict[str, Any]):
    """
    A dict to represent recipe arguments, that can be evaluated
    and used to override values in a recipe

    An evaluated RecipeArgs instance does not contain any eval
    strings as values

    Create and evaluate a RecipeArgs instance:
    >>> recipe_args = RecipeArgs(a="eval(2 * 3)", b=2, c=3)
    >>> recipe_args.evaluate()
    {'a': 6.0, 'b': 2, 'c': 3}


    Create and evaluate a RecipeArgs instance with a parent:
    >>> recipe_args = RecipeArgs(a="eval(x * 3)", b=2, c=3)
    >>> recipe_args.evaluate({"x": 3})
    {'a': 9.0, 'b': 2, 'c': 3}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evaluated: "Optional[RecipeArgs]" = None

    def combine(self, other: Union["RecipeArgs", Dict[str, Any]]) -> "RecipeArgs":
        """
        Helper to combine current recipe args with another set of RecipeArgs
        or a dict

        Combine with another RecipeArgs instance:
        >>> RecipeArgs(a=1, b=2, c=3).combine(RecipeArgs(d=4, e=5))
        {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}

        :param other: The other recipe args or dict to combine with the current
            RecipeArgs instance
        :return: The combined recipe args
        """
        combined = RecipeArgs()
        combined.update(self)

        if other:
            for key in other.keys():
                if not isinstance(key, str):
                    raise ValueError(
                        "`other` must be a RecipeArgs instance or dict with str keys"
                        f" but got {key=} of type {type(key)}"
                    )
            combined.update(other)

        return combined

    def evaluate(self, parent: Optional["RecipeArgs"] = None) -> "RecipeArgs":
        """
        Evaluate the current recipe args and return a new RecipeArgs instance
        with the evaluated values. Can also provide a parent RecipeArgs instance
        to combine with the current instance before evaluating.

        Evaluate with a parent:
        >>> RecipeArgs(a="eval(2 * 3)", b=2).evaluate(
        ... parent=RecipeArgs(c="eval(a * b)")
        ... )
        {'a': 6.0, 'b': 2, 'c': 12.0}

        :param parent: Optional extra recipe args to combine with the current
            instance before evaluating
        :return: The evaluated recipe args
        """
        self._evaluated = RecipeArgs.eval_args(self.combine(parent))

        return self._evaluated


if __name__ == "__main__":
    import doctest

    doctest.testmod()
