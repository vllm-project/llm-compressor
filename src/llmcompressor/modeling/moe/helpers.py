import ast
import inspect
from abc import ABC
from typing import Callable, ClassVar, Optional

import torch
from transformers import PreTrainedConfig
from transformers.core_model_loading import (
    WeightConverter,
    WeightTransform,
)


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


def _is_moe_experts_converter(converter: WeightTransform) -> bool:
    return isinstance(converter, WeightConverter) and converter.target_patterns in (
        ".experts.gate_up_proj",
        ".experts.down_proj",
    )


def _get_moe_shapes(experts: FusedExpertsModule) -> tuple[int, int, int]:
    # get shapes from the down_proj. This is more reliable than getting from config
    return experts.config.n_routed_experts, experts.config.moe_intermediate_size, experts.config.hidden_size
