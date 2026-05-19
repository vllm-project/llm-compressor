import ast
import inspect
from abc import ABC
from typing import Any, Callable, ClassVar, Optional, runtime_checkable

import torch
from transformers import PreTrainedConfig
from transformers.core_model_loading import (
    WeightConverter,
    WeightTransform,
)

from llmcompressor.sentinel import Sentinel
from loguru import logger
from typing import Callable, ClassVar, Optional, Protocol

import torch
from transformers import PreTrainedConfig


@runtime_checkable
class FusedExpertsProtocol(Protocol):
    config: PreTrainedConfig
    has_gate: bool
    has_bias: bool
    is_transposed: bool

    _apply_gate: ClassVar[Callable]

    gate_up_proj: torch.nn.Parameter
    down_proj: torch.nn.Parameter
    act_fn: torch.nn.Module

    up_proj: Optional[torch.nn.Parameter]
    up_proj_bias: Optional[torch.nn.Parameter]
    gate_up_proj_bias: Optional[torch.nn.Parameter]
    down_proj_bias: Optional[torch.nn.Parameter]


def _is_moe_experts_module(module) -> bool:
    """Detect modules whose class is decorated with
    ``@use_experts_implementation`` by inspecting the class source AST."""
    try:
        source = inspect.getsource(type(module))
        tree = ast.parse(source)
    except (OSError, TypeError):
        return False

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                name = decorator.id
            elif isinstance(decorator, ast.Call) and isinstance(
                decorator.func, ast.Name
            ):
                name = decorator.func.id
            else:
                continue
            if name == "use_experts_implementation":
                return True

    return False


def get_use_experts_implementation_args(experts_cls: type) -> dict[str, bool] | None:
    default_args = {
        "is_concatenated": True,
        "is_transposed": False,
        "has_bias": False,
        "has_gate": True,
    }

    try:
        source = inspect.getsource(experts_cls)
        tree = ast.parse(source)
    except (OSError, TypeError):
        logger.warning(f"Could not find source module code for {experts_cls.__name__}")
        return None

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        for decorator in node.decorator_list:
            # Handle @use_experts_implementation (no arguments)
            if (
                isinstance(decorator, ast.Name)
                and decorator.id == "use_experts_implementation"
            ):
                return default_args

            # Handle @use_experts_implementation(...) with arguments
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                if decorator.func.id == "use_experts_implementation":
                    args = default_args

                    # Extract keyword arguments
                    for keyword in decorator.keywords:
                        if keyword.arg in args:
                            # Evaluate the constant value
                            if isinstance(keyword.value, ast.Constant):
                                args[keyword.arg] = keyword.value.value
                            elif isinstance(keyword.value, (ast.Constant, ast.Name)):
                                # Handle True/False/None
                                if isinstance(keyword.value, ast.Constant):
                                    args[keyword.arg] = keyword.value.value
                                elif hasattr(keyword.value, "id"):
                                    # Try to evaluate simple names like True, False
                                    if keyword.value.id in ("True", "False"):
                                        args[keyword.arg] = keyword.value.id == "True"

                    return args

    return None


def get_moe_dims(config: PreTrainedConfig) -> tuple[int, int, int, bool, str, float]:
    return (
        _getattr_fallbacks(config, ["num_local_experts", "moe_num_experts"]),
        _getattr_fallbacks(config, ["hidden_size", "hidden_dim"]),
        _getattr_fallbacks(config, ["moe_intermediate_size"]),
        _getattr_fallbacks(config, ["use_bias", "mlp_bias"]),
        _getattr_fallbacks(config, ["hidden_act"]),
        _getattr_fallbacks(config, ["swiglu_limit"]),
    )


def _getattr_fallbacks(
    target: object, attrs: list[str], default: Any = Sentinel("None")
) -> Any:
    for attr in attrs:
        if hasattr(target, attr):
            return getattr(target, attr)

    if default is not Sentinel("None"):
        return default

    raise AttributeError(f"{target} does not have any of {attrs} attributes")
