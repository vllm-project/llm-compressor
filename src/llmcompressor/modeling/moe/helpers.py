import ast
import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
)

import torch
from compressed_tensors.offload import disable_onloading
from loguru import logger
from transformers import PreTrainedConfig

from llmcompressor.sentinel import Sentinel
from llmcompressor.typing import TorchModuleProtocol


class FusedExpertsProtocol(TorchModuleProtocol):
    config: PreTrainedConfig
    has_gate: bool
    has_bias: bool
    is_transposed: bool
    _apply_gate: ClassVar[Callable]

    down_proj: torch.nn.Parameter
    act_fn: torch.nn.Module

    # gate proj may or may not exist
    gate_up_proj: torch.nn.Parameter
    up_proj: torch.nn.Parameter

    # biases may or may not exist
    up_proj_bias: torch.nn.Parameter
    gate_up_proj_bias: torch.nn.Parameter
    down_proj_bias: torch.nn.Parameter

    @classmethod
    def __validate__(cls, object: object) -> bool:
        with disable_onloading():
            return (
                isinstance(getattr(object, "down_proj", None), torch.nn.Parameter)
                and (
                    isinstance(getattr(object, "up_proj", None), torch.nn.Parameter)
                    or isinstance(
                        getattr(object, "gate_up_proj", None), torch.nn.Parameter
                    )
                )
                and get_use_experts_implementation_args(object.__class__) is not None
            )


def get_use_experts_implementation_args(experts_cls: type) -> dict[str, bool] | None:
    """
    Get the keyword arguments to the `@use_experts_implementation` decorator which
    wraps the class. See `transformers.integrations.moe::use_experts_implementation`.

    :param experts_cls: module class which is decorated by `@use_experts_implementation`
    :return: keyword arguments to decorator, None if the class is not decorated
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
                            if isinstance(keyword.value, ast.Constant):
                                args[keyword.arg] = keyword.value.value
                            elif isinstance(keyword.value, ast.Name):
                                # Handle True/False as Name nodes (older Python AST)
                                if keyword.value.id in ("True", "False"):
                                    args[keyword.arg] = keyword.value.id == "True"
                            else:
                                raise ValueError(f"Invalid keyword {keyword.value}")

                    return args

    return None


@dataclass
class MoEConfig:
    num_experts: int
    num_experts_per_tok: int
    hidden_dim: int
    intermediate_size: int
    use_bias: bool
    hidden_act: str
    limit: float
    alpha: float | None
    dtype: torch.dtype

    @classmethod
    def from_config(cls, config: PreTrainedConfig):
        ret = cls(
            num_experts=_getattr_fallbacks(
                config,
                [
                    "num_local_experts",
                    "moe_num_experts",
                    "num_experts",
                    "n_routed_experts",
                ],
            ),
            num_experts_per_tok=_getattr_fallbacks(
                config, ["top_k_experts", "num_experts_per_tok"]
            ),
            hidden_dim=_getattr_fallbacks(config, ["hidden_size", "hidden_dim"]),
            intermediate_size=_getattr_fallbacks(
                config,
                ["moe_intermediate_size", "intermediate_dim", "intermediate_size"],
            ),
            use_bias=_getattr_fallbacks(config, ["use_bias", "mlp_bias"], False),
            hidden_act=_getattr_fallbacks(
                config, ["hidden_act", "hidden_activation", "mlp_hidden_act"]
            ),
            limit=_getattr_fallbacks(config, ["swiglu_limit"], None),
            alpha=None,
            dtype=_getattr_fallbacks(config, ["dtype"]),
        )

        # special case: GptOssConfig has some parameters which are not marked in config
        if config.model_type == "gpt_oss":
            ret.use_bias = True
            ret.limit = 7.0
            ret.alpha = 1.702

        return ret


def _getattr_fallbacks(
    target: object, attrs: list[str], default: Any = Sentinel("None")
) -> Any:
    for attr in attrs:
        if hasattr(target, attr):
            return getattr(target, attr)

    if default is not Sentinel("None"):
        return default

    raise AttributeError(f"{target} does not have any of {attrs} attributes")
