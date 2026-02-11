"""
Early validation for divisibility requirements by quantization strategy.

Different kernels support different divisibility rules. This module encodes
which strategies require strict divisibility (and thus error early with layer
names) vs which do not.

Policy (single source of truth for "error vs warn vs skip"):

- GROUP, TENSOR_GROUP: Runtime/save kernels require columns % group_size == 0.
  We ERROR at initialize with the list of affected layer FQNs so users can add
  them to `ignore` before long calibration (e.g. GPTQ). No kernel support for
  non-divisible today.

- BLOCK: Block kernels support non-divisible dimensions (e.g. strategy_cdiv
  with strict=False). We do NOT check or warn for block.

- CHANNEL, TENSOR, TOKEN, ATTN_HEAD: No group_size divisibility requirement;
  we do not run this validation.

See: compressed-tensors forward.py (GROUP/TENSOR_GROUP ValueError),
strategy_cdiv in compressed_tensors.quantization.utils.helpers.
"""

from __future__ import annotations

from typing import Set, Tuple

import torch
from compressed_tensors.quantization import QuantizationScheme, QuantizationStrategy
from compressed_tensors.utils import match_named_modules

__all__ = [
    "_layer_indivisible",
    "get_layers_indivisible_by_group_size",
    "validate_group_size_divisibility",
]

# Strategies for which we error on indivisible columns (no kernel support).
# BLOCK is intentionally excluded: block kernels support non-divisible.
_GROUP_STRATEGY_STRINGS = ("group", "tensor_group")


def _is_group_or_tensor_group_strategy(strategy) -> bool:
    """True if strategy is GROUP or TENSOR_GROUP (enum or string)."""
    if strategy is None:
        return False
    if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
        return True
    for attr in ("value", "name"):
        s = getattr(strategy, attr, None)
        if s is not None and str(s).lower() in _GROUP_STRATEGY_STRINGS:
            return True
    s = str(strategy).lower()
    if s in _GROUP_STRATEGY_STRINGS:
        return True
    # Enum repr e.g. "quantizationstrategy.group"
    if s.split(".")[-1] in _GROUP_STRATEGY_STRINGS:
        return True
    return False


def _layer_indivisible(module: torch.nn.Module, weight_args) -> Tuple[int, int] | None:
    """
    If module has group/tensor_group weight and columns % group_size != 0,
    return (columns, group_size); else return None.
    """
    if not _is_group_or_tensor_group_strategy(getattr(weight_args, "strategy", None)):
        return None
    group_size = getattr(weight_args, "group_size", None)
    if group_size is None:
        return None
    if not hasattr(module, "weight"):
        return None
    columns = int(module.weight.shape[-1])
    group_size = int(group_size)
    if columns >= group_size and columns % group_size != 0:
        return (columns, group_size)
    return None


def get_layers_indivisible_by_group_size(
    model: torch.nn.Module,
    resolved_targets: Set[str],
    ignore: list[str],
) -> list[Tuple[str, int, int]]:
    """
    Find targeted layers whose weight columns are not divisible by group_size.

    Only considers layers whose weight scheme is GROUP or TENSOR_GROUP
    (by value; enum or string). BLOCK and other strategies are not checked.
    Matches the condition
    that triggers ValueError in compressed_tensors forward.py (columns >=
    group_size and columns % group_size != 0).

    :param model: Model with quantization schemes already applied (e.g. after
        apply_quantization_config).
    :param resolved_targets: Target module name patterns (e.g. from
        QuantizationMixin.resolved_targets).
    :param ignore: Module name patterns to exclude (e.g. QuantizationMixin.ignore).
    :return: List of (fqn, columns, group_size) for each layer that would
        fail at save/forward due to indivisibility.
    """
    indivisible: list[Tuple[str, int, int]] = []
    for name, module in match_named_modules(model, resolved_targets, ignore):
        scheme: QuantizationScheme | None = getattr(module, "quantization_scheme", None)
        if scheme is None or scheme.weights is None:
            continue
        result = _layer_indivisible(module, scheme.weights)
        if result is not None:
            columns, group_size = result
            indivisible.append((name, columns, group_size))
    return indivisible


def validate_group_size_divisibility(
    model: torch.nn.Module,
    resolved_targets: Set[str],
    ignore: list[str],
    *,
    bypass: bool = False,
) -> None:
    """
    Ensure targeted group/tensor_group layers have columns divisible by group_size.

    If any such layer has columns % group_size != 0, raises ValueError with layer FQNs.
    When bypass is True, skips the check (e.g. for runtimes that support non-divisible).
    """
    if bypass:
        return
    indivisible = get_layers_indivisible_by_group_size(model, resolved_targets, ignore)
    if not indivisible:
        return
    lines = [
        f"  - {fqn} (columns={cols}, group_size={gs})" for fqn, cols, gs in indivisible
    ]
    raise ValueError(
        "The following layers have weight column counts not divisible by "
        "group_size. Group and tensor-group quantization require "
        "columns % group_size == 0; compressed-tensors will error when saving "
        "or running forward. Add these layer names to the modifier's `ignore` "
        "list and re-run, or set bypass_divisibility_checks=True if your "
        "runtime (e.g. vLLM) supports non-divisible dimensions.\n\n" + "\n".join(lines)
    )
