from __future__ import annotations

import ast
import importlib
import inspect
from collections import ChainMap
from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import torch
from compressed_tensors.offload import disable_onloading
from loguru import logger
from transformers import PreTrainedConfig

from llmcompressor.typing import TorchModuleProtocol
from llmcompressor.utils import hasattr_fallbacks

# MoEConfig fields which are read from a config dict, paired with the
# candidate keys (in fallback order) each is searched under. The key a value
# was read from is recorded in the field's ``<field>_key`` sibling attribute
# (e.g. ``num_experts_key``), so tooling can write updated values back to the
# same key
MOE_CONFIG_KEYS = {
    "num_experts": [
        "num_local_experts",
        "moe_num_experts",
        "num_experts",
        "n_routed_experts",
    ],
    "num_experts_per_tok": [
        "top_k_experts",
        "num_experts_per_tok",
        "num_experts_per_token",
    ],
    # group-limited routing (e.g. DeepSeek-V3): experts are partitioned into
    # ``n_group`` contiguous groups of which only the top ``top_k_group`` are
    # routable per token
    "n_group": [
        "n_group",
        "num_expert_group",
        "n_expert_groups",
    ],
    "top_k_group": [
        "topk_group",
        "top_k_group",
        "n_limited_groups",
    ],
    "hidden_dim": [
        "moe_latent_size",
        "hidden_size",
        "hidden_dim",
    ],
    "intermediate_size": [
        "moe_intermediate_size",
        "intermediate_dim",
        "intermediate_size",
    ],
    "use_bias": [
        "use_bias",
        "mlp_bias",
    ],
    "hidden_act": [
        "hidden_act",
        "hidden_activation",
        "mlp_hidden_act",
    ],
    "limit": [
        "swiglu_limit",
    ],
    "dtype": [
        "dtype",
        "torch_dtype",
    ],
}

# config fields which declare a model's native router score transformation
# (the transformation a router applies to its logits to produce the
# nonnegative scores which select and weight the experts), checked in this
# order when resolving ``score_function``; the declared value is kept as-is and
# falls back to ``softmax`` when no field is present
SCORE_FUNCTION_CONFIG_KEYS = [
    "scoring_func",
    "score_func",
    "moe_router_activation_func",
]


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
    # experts routing
    num_experts: int | None = None
    num_experts_key: str | None = None
    num_experts_per_tok: int | None = None
    num_experts_per_tok_key: str | None = None

    # router scoring
    score_function: str = "softmax"
    score_function_key: str | None = None

    # group-limited routing
    n_group: int | None = None
    n_group_key: str | None = None
    top_k_group: int | None = None
    top_k_group_key: str | None = None
    group_size: int | None = None

    # hidden sizes
    hidden_dim: int | None = None  # input/output width of each expert
    hidden_dim_key: str | None = None
    intermediate_size: int | None = None
    intermediate_size_key: str | None = None

    # biases and activations
    use_bias: bool = False
    use_bias_key: str | None = None
    hidden_act: str | None = None
    hidden_act_key: str | None = None

    # moe extras
    limit: float | None = None
    limit_key: str | None = None
    alpha: float | None = None

    # weight dtypes
    dtype: torch.dtype | None = None
    dtype_key: str | None = None

    @classmethod
    def from_config(cls, config: PreTrainedConfig) -> MoEConfig:
        """
        Build a ``MoEConfig`` from a ``PreTrainedConfig`` by converting it to
        a dict with ``to_dict`` and delegating to :meth:`from_dict`.

        Note that a config serializes under its canonical key names, which may
        differ from its attribute names (e.g. a ``Qwen3MoeConfig`` built with
        ``num_experts=4`` reports it under ``num_local_experts``). The
        candidate key lists cover these canonical names.
        """
        return cls.from_dict(config.to_dict())

    @classmethod
    def from_dict(cls, config_data: dict) -> MoEConfig:
        """
        Build a ``MoEConfig`` from a raw config dict, as loaded from a
        checkpoint's config.json.

        Every field is optional: values which the dict does not declare are
        left ``None`` (or ``False``), since a config dict may omit fields an
        in-memory model config carries. Each field is searched under its
        candidate keys (``MOE_CONFIG_KEYS``) at the top level of the dict and
        then under its nested language-model config, supporting composite (e.g.
        vision-language) checkpoints which nest it under ``text_config`` or
        ``llm_config`` (e.g. nemotron_h_omni).

        The key each value was read from is recorded in the field's
        ``<field>_key`` sibling attribute (e.g. ``num_experts_key``), so
        tooling can write updated values back to the same key.
        """
        # composite checkpoints keep their text-model fields under a nested
        # config; the top level takes precedence for a given key, then
        # text_config, then llm_config
        nested = (config_data.get("text_config"), config_data.get("llm_config"))
        config = ChainMap(
            config_data, *(cfg for cfg in nested if isinstance(cfg, dict))
        )

        # find the key each value is stored under, then read the value
        # through that key
        kwargs: dict[str, Any] = {}
        for field, keys in MOE_CONFIG_KEYS.items():
            key = hasattr_fallbacks(config, keys, return_attr=True)
            kwargs[f"{field}_key"] = key or None
            kwargs[field] = config[key] if key else None
        self = cls(**kwargs)

        # resolve the transform the modifiers apply to a router's captured
        # output to recover its per-expert scores. For routers whose forward
        # returns raw logits this is their native score function (``softmax``
        # by default, read from the config here); routers which already return
        # scored probabilities are handled as ``identity`` in the special cases
        # below
        for key in SCORE_FUNCTION_CONFIG_KEYS:
            value = config.get(key)
            if isinstance(value, str):
                self.score_function = value
                self.score_function_key = key
                break

        self.use_bias = bool(self.use_bias)
        self.dtype = _to_torch_dtype(self.dtype)
        self._resolve_group_size()

        # special case: some configs have parameters which are not marked in
        # config
        match config_data.get("model_type"):
            case "gpt_oss" | "openai_privacy_filter":
                self.use_bias = True
                self.limit = 7.0
                self.alpha = 1.702
                self.hidden_act = "sigmoid"
            case "lfm2_moe":
                self.hidden_act = "silu"
            case "nemotron_h":
                # the router hardcodes a sigmoid score function (applied to the
                # logits its forward returns); the config carries no scoring
                # field. Checkpoints with a moe_latent_size project the block
                # input down before the experts, which the hidden-size fallback
                # already resolves into hidden_dim (the experts' width)
                self.score_function = "sigmoid"
                # moe_latent_size may be declared but unset; the expert hidden
                # dim is then the model hidden size
                if self.hidden_dim is None:
                    self.hidden_dim_key = "hidden_size"
                    self.hidden_dim = config.get("hidden_size")
            case "gemma4" | "gemma4_text":
                # the router's forward already returns softmaxed probabilities
                # (not logits), so its captured output is the per-expert scores
                # directly; no further score transform is applied
                self.score_function = "identity"

        return self

    def _resolve_group_size(self):
        """Compute ``group_size`` from ``num_experts`` and ``n_group`` when both
        are known, raising if the experts do not split evenly into groups."""
        if self.n_group == 1:
            # a single routing group is no group limitation at all: normalize
            # it away so the group-limited code paths (equal drops per group,
            # per-group saliency ranking) are not taken spuriously (e.g.
            # nemotron_h, whose config declares n_group=1)
            self.n_group = None
            self.n_group_key = None
            self.top_k_group = None
            self.top_k_group_key = None
        if self.n_group is None or self.num_experts is None:
            return
        if self.num_experts % self.n_group != 0:
            raise ValueError(
                f"Group limited router detected, but {self.num_experts} experts "
                f"not divisible by n_group={self.n_group}"
            )
        self.group_size = self.num_experts // self.n_group


def import_or_none(paths: str | list[str]) -> object | None:
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        try:
            module_path, attr_name = path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, attr_name)
        except (ImportError, AttributeError):
            pass

    return None


def _to_torch_dtype(value: Any) -> torch.dtype | None:
    """
    Convert a config dtype value to a ``torch.dtype``. Config dicts carry
    dtype names as strings (``"bfloat16"`` or ``"torch.bfloat16"``); values
    which are already dtypes pass through, anything else maps to ``None``.
    """
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        return getattr(torch, value.removeprefix("torch."), None)
    return None
