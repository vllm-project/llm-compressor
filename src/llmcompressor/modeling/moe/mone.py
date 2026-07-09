"""
Helpers for loading and running MoNE-pruned MoE models.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from compressed_tensors import align_module_device
from loguru import logger

from llmcompressor.modeling.moe.linear_experts import LinearExperts2D, NoviceExpertMLP

__all__ = [
    "MoNEModelSupport",
    "apply_mone_structure",
    "get_mone_model_processor_source",
    "is_mone_checkpoint",
    "load_mone_checkpoint",
    "postprocess_mone_export",
    "prepare_model_for_mone",
    "prepare_mone_model_for_save",
    "register_mone_model_support",
]


ROUTER_ATTRS = ("router", "gate")
EXPERTS_ATTRS = ("experts",)
_MONE_MODEL_SUPPORTS: dict[str, "MoNEModelSupport"] = {}
_BUILTIN_SUPPORT_LOADED = False


@dataclass(frozen=True)
class MoNEModelSupport:
    name: str
    prepare_model: Callable[[nn.Module], list[str] | None] | None = None
    prepare_for_save: Callable[[nn.Module], None] | None = None
    postprocess_export: Callable[[nn.Module, str | Path], None] | None = None
    is_checkpoint: Callable[[str | Path], bool] | None = None
    load_checkpoint: Callable[..., nn.Module] | None = None
    processor_source: Callable[[nn.Module], str | Path | None] | None = None


def register_mone_model_support(support: MoNEModelSupport) -> None:
    """
    Register architecture-specific MoNE hooks.
    """

    _MONE_MODEL_SUPPORTS[support.name] = support


def prepare_model_for_mone(model: nn.Module) -> list[str]:
    """
    Apply architecture-specific runtime preparation needed before MoNE calibration.

    The generic MoNE modifier should not need to know which model family needs
    which patch. Family-specific support is dispatched from here.
    """

    patches = []
    for support in _iter_mone_model_supports():
        if support.prepare_model is None:
            continue
        patches.extend(support.prepare_model(model) or [])
    return patches


def is_mone_checkpoint(model_path: str | Path) -> bool:
    """
    Return True when ``model_path`` is a MoNE checkpoint with custom load support.
    """

    return any(
        support.is_checkpoint is not None and support.is_checkpoint(model_path)
        for support in _iter_mone_model_supports()
    )


def load_mone_checkpoint(model_path: str | Path, **load_kwargs) -> nn.Module:
    """
    Load a MoNE checkpoint using the first registered matching checkpoint loader.
    """

    for support in _iter_mone_model_supports():
        if support.is_checkpoint is None or not support.is_checkpoint(model_path):
            continue
        if support.load_checkpoint is None:
            raise ValueError(
                f"MoNE support '{support.name}' recognized {model_path} but does "
                "not provide a checkpoint loader."
            )
        return support.load_checkpoint(model_path, **load_kwargs)

    raise ValueError(f"No registered MoNE checkpoint loader recognized {model_path}")


def get_mone_model_processor_source(model: nn.Module) -> str | Path | None:
    """
    Return a tokenizer/processor source associated with a loaded MoNE model.
    """

    for support in _iter_mone_model_supports():
        if support.processor_source is None:
            continue
        source = support.processor_source(model)
        if source is not None:
            return source
    return None


def prepare_mone_model_for_save(model: nn.Module) -> None:
    """
    Attach architecture-specific export metadata after MoNE has modified a model.

    Generic linearized MoE models can be saved as-is. Architectures with custom
    checkpoint layouts can register their export preparation here.
    """

    for support in _iter_mone_model_supports():
        if support.prepare_for_save is not None:
            support.prepare_for_save(model)


def postprocess_mone_export(model: nn.Module, output_dir: str | Path) -> None:
    """
    Post-process a saved MoNE checkpoint for architecture-specific layouts.
    """

    for support in _iter_mone_model_supports():
        if support.postprocess_export is not None:
            support.postprocess_export(model, output_dir)


def _iter_mone_model_supports() -> tuple[MoNEModelSupport, ...]:
    _load_builtin_mone_support()
    return tuple(_MONE_MODEL_SUPPORTS.values())


def _load_builtin_mone_support() -> None:
    global _BUILTIN_SUPPORT_LOADED

    if _BUILTIN_SUPPORT_LOADED:
        return

    _BUILTIN_SUPPORT_LOADED = True
    try:
        from llmcompressor.modeling.moe.mone_builtins import (
            load_builtin_mone_support,
        )

        load_builtin_mone_support()
    except Exception:
        _BUILTIN_SUPPORT_LOADED = False
        raise


def apply_mone_structure(
    model: nn.Module,
    ignore: list[str] | None = None,
    strict: bool = True,
) -> dict[str, list[int]]:
    """
    Replace configured MoNE novice expert slots with ``NoviceExpertMLP`` modules.

    This is intended to run after model construction and before loading a MoNE
    state dict. A normal MoE constructor creates full MLP experts for every
    expert id, while a MoNE checkpoint stores ``approx_value`` for novice expert
    ids. Applying this structure makes strict state-dict loading possible.

    :param model: MoE model whose config contains ``approximate_experts``.
    :param ignore: optional regex patterns for MoE layer names to skip.
    :param strict: raise if config references a layer key not found in the model.
    :return: mapping from matched MoE layer names to novice expert indices.
    """

    approximate_experts, approximate_tokens = _mone_config_maps(model)
    if not approximate_experts:
        return {}

    ignored = ignore or []
    matched_config_keys: set[str] = set()
    replaced: dict[str, list[int]] = {}

    for layer_name, experts in _iter_linear_moe_experts(model, ignored):
        config_key = _config_layer_key(layer_name)
        novice_indices = _lookup_layer_values(approximate_experts, layer_name)
        if novice_indices is None:
            continue

        matched_config_keys.add(config_key)
        matched_config_keys.add(layer_name)

        token_values = _lookup_layer_values(approximate_tokens, layer_name) or []
        token_by_expert = {
            expert_idx: int(token_values[pos])
            for pos, expert_idx in enumerate(novice_indices)
            if pos < len(token_values)
        }

        for expert_idx in novice_indices:
            _replace_expert_with_novice(
                experts=experts,
                expert_idx=expert_idx,
                acc_tokens=token_by_expert.get(expert_idx, 0),
            )

        replaced[layer_name] = novice_indices

    if strict:
        unmatched = set(approximate_experts) - matched_config_keys
        if unmatched:
            raise ValueError(
                "MoNE config references layers that were not found in the model: "
                f"{sorted(unmatched)}"
            )

    if replaced:
        logger.info(f"Applied MoNE novice expert structure to {len(replaced)} layers")

    return replaced


def _iter_linear_moe_experts(
    model: nn.Module,
    ignore: list[str],
) -> list[tuple[str, LinearExperts2D]]:
    layers = []
    for name, module in model.named_modules():
        if any(re.search(pattern, name) for pattern in ignore):
            continue
        if not any(hasattr(module, router_attr) for router_attr in ROUTER_ATTRS):
            continue

        for experts_attr in EXPERTS_ATTRS:
            experts = getattr(module, experts_attr, None)
            if isinstance(experts, LinearExperts2D):
                layers.append((name, experts))
                break

    return layers


def _replace_expert_with_novice(
    experts: LinearExperts2D,
    expert_idx: int,
    acc_tokens: int,
):
    if expert_idx < 0 or expert_idx >= experts.num_experts:
        raise ValueError(
            f"MoNE novice expert index {expert_idx} is out of range for "
            f"num_experts={experts.num_experts}"
        )

    old_expert = experts[expert_idx]
    if isinstance(old_expert, NoviceExpertMLP):
        old_expert.acc_tokens = acc_tokens
        return

    hidden_size = _expert_hidden_size(old_expert)
    with align_module_device(old_expert):
        dtype = _first_floating_dtype(old_expert, torch.float32)
        device = _first_device(old_expert, torch.device("cpu"))

    experts[expert_idx] = NoviceExpertMLP(
        hidden_dim=hidden_size,
        dtype=dtype,
        acc_tokens=acc_tokens,
    ).to(device)


def _mone_config_maps(
    model: nn.Module,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    for config in _candidate_configs(model):
        approximate_experts = getattr(config, "approximate_experts", None)
        if not approximate_experts:
            continue

        approximate_tokens = getattr(config, "approximate_expert_init_tokens", None)
        return _normalize_layer_map(approximate_experts), _normalize_layer_map(
            approximate_tokens or {}
        )

    return {}, {}


def _candidate_configs(model: nn.Module) -> list[object]:
    config = getattr(model, "config", None)
    if config is None:
        return []

    configs = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        configs.append(text_config)
    return configs


def _normalize_layer_map(values: dict | object) -> dict[str, list[int]]:
    if not isinstance(values, dict):
        return {}

    normalized: dict[str, list[int]] = {}
    for key, layer_values in values.items():
        if layer_values is None:
            normalized[str(key)] = []
        else:
            normalized[str(key)] = [int(value) for value in layer_values]
    return normalized


def _lookup_layer_values(
    values: dict[str, list[int]],
    layer_name: str,
) -> list[int] | None:
    config_key = _config_layer_key(layer_name)
    if config_key in values:
        return values[config_key]
    if layer_name in values:
        return values[layer_name]
    return None


def _config_layer_key(layer_name: str) -> str:
    parts = layer_name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers" and parts[idx + 1].isdigit():
            return parts[idx + 1]
    return layer_name


def _expert_hidden_size(module: nn.Module) -> int:
    down_proj = getattr(module, "down_proj", None)
    if isinstance(down_proj, nn.Linear):
        return down_proj.out_features

    for param in module.parameters(recurse=True):
        if param.ndim > 0:
            return int(param.shape[-1])

    raise ValueError(f"Could not infer hidden size for expert {module}")


def _first_device(module: nn.Module, fallback: torch.device) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return fallback


def _first_floating_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    for param in module.parameters(recurse=True):
        if param.is_floating_point():
            return param.dtype
    for buffer in module.buffers(recurse=True):
        if buffer.is_floating_point():
            return buffer.dtype
    return fallback
