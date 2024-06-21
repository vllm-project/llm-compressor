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

    def evaluate_ext(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate external target dict with the current recipe args and return
        the resulting evaluated dict

        Evaluate target with values from current recipe args:
        >>> RecipeArgs(a=6).evaluate_ext({"b": "eval(a * 2)"})
        {'b': 12.0}

        :param target: The target dict to evaluate
        :return: The combined RecipeArgs instance with the evaluated values
        """
        args = RecipeArgs.eval_args(self)
        resolved = {}

        for key, value in target.items():
            resolved[key] = RecipeArgs.eval_obj(value, args)

        return resolved

    @staticmethod
    def eval_str(
        target: str, args: Optional[Dict[str, Any]] = None
    ) -> Union[str, float]:
        """
        Evaluate the target string with the supplied args and return the
        resulting evaluated value

        Evaluate a string with a variable and args:
        >>> RecipeArgs.eval_str("eval(a * 3)", {"a": 2})
        6.0

        :param target: The target string to evaluate
        :param args: The args to use for the evaluation, these will be used
            as the local namespace for the eval
        :return: The evaluated value of the target string
        """
        if "eval(" not in target:
            return target

        pattern = re.compile(r"eval\(([^()]*)\)")
        match = pattern.search(target)

        if not match:
            raise ValueError(f"invalid eval string {target}")

        inner_expr = match.group(1)
        result = eval(inner_expr, {"math": math}, args if args else {})
        new_target = target.replace(match.group(0), str(result))
        try:
            return float(new_target)
        except ValueError:
            return RecipeArgs.eval_str(new_target, args)

    @staticmethod
    def eval_args(args: Dict[str, Any]) -> "RecipeArgs":
        """
        Evaluate supplied args and return a new RecipeArgs instance with the
        evaluated values

        Order of evaluation is automatically handled:
        >>> RecipeArgs.eval_args({"a": "eval(b * 2)", "b": 2, "c": 3})
        {'a': 4.0, 'b': 2, 'c': 3}

        :param args: The args to evaluate, must be a dict of str to any
        :return: The evaluated recipe args instance
        """
        resolved = args.copy()

        while True:
            for key, value in resolved.items():
                if isinstance(value, str):
                    resolved[key] = RecipeArgs.eval_str(value, resolved)
                else:
                    resolved[key] = value

            if args == resolved:
                break
            else:
                args = resolved.copy()

        return RecipeArgs(resolved)

    @staticmethod
    def eval_obj(target: Any, args: Optional[Dict[str, Any]] = None) -> Any:
        """
        Generic version of `eval_str` that can evaluate any object
        recursively, supports str, dict, and list types

        >>> RecipeArgs.eval_obj("eval(a * 3)", {"a": 2})
        6.0

        >>> RecipeArgs.eval_obj({"a": "eval(a * 3)"}, {"a": 2})
        {'a': 6.0}

        >>> RecipeArgs.eval_obj(["eval(a * 3)"], {"a": 2})
        [6.0]
        """
        if isinstance(target, str):
            return RecipeArgs.eval_str(target, args)
        elif isinstance(target, dict):
            return {
                key: RecipeArgs.eval_obj(value, args) for key, value in target.items()
            }
        elif isinstance(target, list):
            return [RecipeArgs.eval_obj(item, args) for item in target]

        return target


if __name__ == "__main__":
    import doctest

    doctest.testmod()
