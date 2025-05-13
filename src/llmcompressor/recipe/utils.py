import re
import math
from typing import Any, Dict, Optional, Union


def evaluate_ext(target: Dict[str, Any], context_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a target dictionary using a context of evaluated arguments.

    >>> evaluate_ext({"b": "eval(a * 2)"}, {"a": 6})
    {'b': 12.0}
    """
    resolved = {}
    for key, value in target.items():
        resolved[key] = eval_obj(value, context_args)
    return resolved


def eval_str(target: str, args: Optional[Dict[str, Any]] = None) -> Union[str, float]:
    """
    Evaluate a string expression with optional arguments.

    >>> eval_str("eval(a * 3)", {"a": 2})
    6.0
    """
    if "eval(" not in target:
        return target

    pattern = re.compile(r"eval\(([^()]*)\)")
    match = pattern.search(target)

    if not match:
        raise ValueError(f"Invalid eval string: {target}")

    inner_expr = match.group(1)
    result = eval(inner_expr, {"math": math}, args or {})
    new_target = target.replace(match.group(0), str(result))

    try:
        return float(new_target)
    except ValueError:
        return eval_str(new_target, args)


def eval_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively evaluate a dictionary of arguments.

    >>> eval_args({"a": "eval(b * 2)", "b": 2, "c": 3})
    {'a': 4.0, 'b': 2, 'c': 3}
    """
    resolved = args.copy()

    while True:
        for key, value in resolved.items():
            if isinstance(value, str):
                resolved[key] = eval_str(value, resolved)
            else:
                resolved[key] = value

        if args == resolved:
            break
        args = resolved.copy()

    return resolved


def eval_obj(target: Any, args: Optional[Dict[str, Any]] = None) -> Any:
    """
    Recursively evaluate strings, dicts, and lists with eval(...) expressions.

    >>> eval_obj("eval(a * 3)", {"a": 2})
    6.0
    """
    if isinstance(target, str):
        return eval_str(target, args)
    elif isinstance(target, dict):
        return {key: eval_obj(val, args) for key, val in target.items()}
    elif isinstance(target, list):
        return [eval_obj(item, args) for item in target]
    return target
