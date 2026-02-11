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
    "STRATEGIES_REQUIRING_STRICT_GROUP_DIVISIBILITY",
    "get_layers_indivisible_by_group_size",
]

# Strategies for which we error on indivisible columns (no kernel support).
# BLOCK is intentionally excluded: block kernels support non-divisible.
STRATEGIES_REQUIRING_STRICT_GROUP_DIVISIBILITY = (
    QuantizationStrategy.GROUP,
    QuantizationStrategy.TENSOR_GROUP,
)


def get_layers_indivisible_by_group_size(
    model: torch.nn.Module,
    resolved_targets: Set[str],
    ignore: list[str],
) -> list[Tuple[str, int, int]]:
    """
    Find targeted layers whose weight columns are not divisible by group_size.

    Only considers layers whose weight scheme is in
    STRATEGIES_REQUIRING_STRICT_GROUP_DIVISIBILITY (GROUP, TENSOR_GROUP).
    BLOCK and other strategies are not checked. Matches the condition
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
        args = scheme.weights
        if args.strategy not in STRATEGIES_REQUIRING_STRICT_GROUP_DIVISIBILITY:
            continue
        group_size = getattr(args, "group_size", None)
        if group_size is None:
            continue
        if not hasattr(module, "weight"):
            continue
        weight = module.weight
        # Same "columns" as compressed_tensors forward: last dim of weight
        columns = weight.shape[-1]
        if columns >= group_size and columns % group_size != 0:
            indivisible.append((name, columns, group_size))
    return indivisible
