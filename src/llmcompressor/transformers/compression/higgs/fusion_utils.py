"""
Fused layer detection for quantization scheme assignment.

Identifies groups of layers that should use the same quantization scheme:
- MoE expert layers: all experts in the same transformer layer must share a scheme
  (vLLM stacks expert weights into fused tensors)
- MLP layers: gate_proj+up_proj (or w1+w3)
- Attention layers: q_proj+k_proj+v_proj
- DeepSeek MQA: q_a_proj+kv_a_proj_with_mqa
"""

import re
from collections import defaultdict
from typing import Dict, List, Set

from llmcompressor.observers.helpers import FUSED_LAYER_NAMES
from loguru import logger


__all__ = ["detect_fused_groups"]

_EXPERT_PATTERN = re.compile(r"(.+\.experts)\.\d+\..+")


def _detect_moe_expert_groups(
    layer_names: List[str],
) -> tuple[List[List[str]], Set[str]]:
    """
    Group all MoE expert layers within the same transformer layer.

    vLLM stacks all experts into fused tensors (w13_weight for gate+up,
    w2_weight for down), so all experts in a layer must use the same
    quantization scheme.

    Returns:
        Tuple of (groups, processed_names) where groups is a list of
        expert fusion groups and processed_names tracks which layers
        have been assigned to a group.
    """
    expert_groups: Dict[str, List[str]] = defaultdict(list)

    for name in layer_names:
        m = _EXPERT_PATTERN.match(name)
        if m:
            expert_groups[m.group(1)].append(name)

    groups = []
    processed = set()
    for prefix, members in sorted(expert_groups.items()):
        if len(members) > 1:
            groups.append(sorted(members))
            processed.update(members)

    if groups:
        total_layers = sum(len(g) for g in groups)
        logger.info(
            f"Detected {len(groups)} MoE expert fusion groups "
            f"({total_layers} expert layers fused)"
        )

    return groups, processed


def detect_fused_groups(layer_names: List[str]) -> List[List[str]]:
    """
    Group layers that should share the same quantization scheme.

    Two fusion mechanisms:
    1. MoE expert fusion: all experts in the same transformer layer are
       grouped together (e.g. all 128 experts' gate/up/down_proj in one group).
    2. Suffix-based fusion from FUSED_LAYER_NAMES: gate+up, qkv, etc.
       Applied to non-MoE layers (attention, shared experts, dense MLP).

    Args:
        layer_names: List of layer names to analyze

    Returns:
        List of groups, where each group is a list of layer names that must
        use the same quantization scheme
    """
    layer_name_set = set(layer_names)

    # Phase 1: MoE expert fusion (all experts in a layer share one scheme)
    moe_groups, processed = _detect_moe_expert_groups(layer_names)
    groups = list(moe_groups)

    # Phase 2: suffix-based fusion for remaining layers (attention qkv, mlp gate+up)
    for layer_name in layer_names:
        if layer_name in processed:
            continue

        current_group = [layer_name]
        processed.add(layer_name)

        for fusion_pattern in FUSED_LAYER_NAMES:
            matching_suffix = None
            for suffix in fusion_pattern:
                if layer_name.endswith(suffix):
                    matching_suffix = suffix
                    break

            if matching_suffix is None:
                continue

            base_name = layer_name[: -len(matching_suffix)]

            for other_suffix in fusion_pattern:
                if other_suffix == matching_suffix:
                    continue

                candidate_name = base_name + other_suffix

                if candidate_name in layer_name_set and candidate_name not in processed:
                    current_group.append(candidate_name)
                    processed.add(candidate_name)

        if len(current_group) > 1:
            groups.append(sorted(current_group))

    logger.info(
        f"Detected {len(groups)} fused layer groups "
        f"(total {sum(len(g) for g in groups)} layers affected)"
    )

    return groups
