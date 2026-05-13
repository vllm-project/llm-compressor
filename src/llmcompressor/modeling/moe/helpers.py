import ast
import inspect
from abc import ABC
from typing import Callable, ClassVar, Optional, Any

import torch
from transformers import PreTrainedConfig
from transformers.activations import ACT2FN
from transformers.core_model_loading import (
    WeightConverter,
    WeightTransform,
)

from llmcompressor.sentinel import Sentinel


class FusedExpertsModule(torch.nn.Module, ABC):
    """
    Fake Typing Class

    """

    config: PreTrainedConfig
    has_gate: bool
    has_bias: bool
    is_transposed: bool
    _apply_gate: ClassVar[Callable]

    gate_up_proj: torch.nn.Parameter
    down_proj: torch.nn.Parameter
    act_fn: torch.nn.Module

    up_proj: Optional[torch.nn.Parameter]  # not has_gate
    up_proj_bias: Optional[torch.nn.Parameter]  # not has_gate, has_bias
    gate_up_proj_bias: Optional[torch.nn.Parameter]  # has_bias
    down_proj_bias: Optional[torch.nn.Parameter]  # has_bias


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


def get_use_experts_implementation_args(experts_cls: type) -> dict[str, bool]:
    """Extract arguments from @use_experts_implementation decorator by inspecting the class source AST.

    Args:
        experts_cls: The experts class decorated with @use_experts_implementation

    Returns:
        Dictionary of decorator arguments (is_concatenated, is_transposed, has_bias, has_gate)
    """
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
        return default_args

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        for decorator in node.decorator_list:
            # Handle @use_experts_implementation (no arguments)
            if isinstance(decorator, ast.Name) and decorator.id == "use_experts_implementation":
                return default_args

            # Handle @use_experts_implementation(...) with arguments
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                if decorator.func.id == "use_experts_implementation":
                    args = default_args.copy()

                    # Extract keyword arguments
                    for keyword in decorator.keywords:
                        if keyword.arg in args:
                            # Evaluate the constant value
                            if isinstance(keyword.value, ast.Constant):
                                args[keyword.arg] = keyword.value.value
                            elif isinstance(keyword.value, (ast.NameConstant, ast.Name)):
                                # Handle True/False/None
                                if isinstance(keyword.value, ast.NameConstant):
                                    args[keyword.arg] = keyword.value.value
                                elif hasattr(keyword.value, 'id'):
                                    # Try to evaluate simple names like True, False
                                    if keyword.value.id in ('True', 'False'):
                                        args[keyword.arg] = keyword.value.id == 'True'

                    return args

    return default_args


def _is_moe_experts_converter(converter: WeightTransform) -> bool:
    return isinstance(converter, WeightConverter) and converter.target_patterns in (
        ".experts.gate_up_proj",
        ".experts.down_proj",
    )


def get_moe_dims(config: PreTrainedConfig) -> tuple[int, int, int, bool, str]:
    return (
        _getattr_fallbacks(config, ["num_local_experts", "moe_num_experts"]),
        _getattr_fallbacks(config, ["hidden_size", "hidden_dim"]),
        _getattr_fallbacks(config, ["moe_intermediate_size"]),
        _getattr_fallbacks(config, ["use_bias", "mlp_bias"]),
        _getattr_fallbacks(config, ["hidden_act"]),
    )


def _getattr_fallbacks(target: object, attrs: list[str], default: Any = Sentinel("None")) -> Any:
    for attr in attrs:
        if hasattr(target, attr):
            return getattr(target, attr)
        
    if default is not Sentinel("None"):
        return default
    
    raise AttributeError(f"{target} does not have any of {attrs} attributes")
    