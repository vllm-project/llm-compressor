"""
Utility functions for HIGGS mixed-precision quantization.

MSE computation, heuristic alpha calculation, fused layer detection,
and config generation from ILP solutions.
"""

import re
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from compressed_tensors.compressors.format import infer_module_format
from compressed_tensors.quantization import (
    QuantizationScheme,
    fake_quantize,
    initialize_module_for_quantization,
)
from loguru import logger

from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    observe,
    update_qparams,
)
from llmcompressor.observers.helpers import FUSED_LAYER_NAMES


__all__ = [
    "compute_layer_mse",
    "compute_heuristic_alphas",
    "generate_config_groups",
    "detect_fused_groups",
]


# ---------------------------------------------------------------------------
# MSE computation
# ---------------------------------------------------------------------------

def compute_layer_mse(weight, scheme, device=None):
    """Compute MSE between original and fake-quantized weight."""
    try:
        device = device or weight.device
        weight = weight.to(device)
        out_features, in_features = weight.shape

        module = torch.nn.Linear(in_features, out_features, bias=False, device="meta")
        module.weight = torch.nn.Parameter(weight, requires_grad=False)

        initialize_module_for_quantization(module, scheme, force_zero_point=False)
        initialize_observer(module, "weight")
        apply_calibration_status(module)
        observe(module, base_name="weight")
        update_qparams(module, base_name="weight")
        freeze_module_quantization(module)

        scale = module.weight_scale
        zp = getattr(module, "weight_zero_point", torch.zeros_like(scale))
        quantized = fake_quantize(x=weight, scale=scale, zero_point=zp, args=scheme.weights)

        return torch.mean((weight - quantized) ** 2).item()

    except Exception as e:
        logger.warning(f"Failed to compute MSE for scheme {scheme}: {e}. Returning inf.")
        return float("inf")


# ---------------------------------------------------------------------------
# Alpha heuristic: alpha = log(size+1) * (1 + depth*0.05) * type_multiplier
# ---------------------------------------------------------------------------

_DEPTH_RE = re.compile(r"(?:layers|layer|h|blocks)\.(\d+)\.")
_TYPE_KEYWORDS = {
    "attention": (["attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj"], 1.2),
    "embedding": (["embed"], 1.5),
    "mlp": (["mlp", "ffn", "fc", "gate_proj", "up_proj", "down_proj"], 0.9),
}


def compute_heuristic_alphas(
    layer_names: List[str],
    layer_param_counts: Dict[str, int],
) -> Dict[str, float]:
    """Compute importance weights: log(size+1) * (1 + depth*0.05) * type_multiplier."""
    alphas = {}
    for name in layer_names:
        size = layer_param_counts.get(name, 0)
        if size == 0:
            alphas[name] = 1.0
            continue

        depth_match = _DEPTH_RE.search(name)
        depth = int(depth_match.group(1)) if depth_match else 0

        type_mult = 1.0
        lower = name.lower()
        for keywords, mult in _TYPE_KEYWORDS.values():
            if any(kw in lower for kw in keywords):
                type_mult = mult
                break

        alphas[name] = float(np.log(size + 1) * (1 + depth * 0.05) * type_mult)

    if alphas:
        vals = list(alphas.values())
        logger.info(
            f"Heuristic alphas for {len(alphas)} layers: "
            f"range=[{min(vals):.2f}, {max(vals):.2f}], mean={np.mean(vals):.2f}"
        )
    return alphas


# ---------------------------------------------------------------------------
# Fused layer detection
# ---------------------------------------------------------------------------

_MOE_EXPERT_RE = re.compile(r"(.+\.experts)\.\d+\..+")


def detect_fused_groups(layer_names: List[str]) -> List[List[str]]:
    """Group layers that must share the same quantization scheme (MoE experts, qkv, gate+up)."""
    layer_name_set = set(layer_names)

    # Phase 1: MoE expert fusion
    expert_groups: Dict[str, List[str]] = defaultdict(list)
    for name in layer_names:
        m = _MOE_EXPERT_RE.match(name)
        if m:
            expert_groups[m.group(1)].append(name)

    groups = []
    processed = set()
    for members in expert_groups.values():
        if len(members) > 1:
            groups.append(sorted(members))
            processed.update(members)

    # Phase 2: suffix-based fusion (attention qkv, mlp gate+up)
    for name in layer_names:
        if name in processed:
            continue
        current_group = [name]
        processed.add(name)
        for fusion_pattern in FUSED_LAYER_NAMES:
            matching_suffix = next((s for s in fusion_pattern if name.endswith(s)), None)
            if matching_suffix is None:
                continue
            base = name[: -len(matching_suffix)]
            for other in fusion_pattern:
                candidate = base + other
                if other != matching_suffix and candidate in layer_name_set and candidate not in processed:
                    current_group.append(candidate)
                    processed.add(candidate)
        if len(current_group) > 1:
            groups.append(sorted(current_group))

    logger.info(f"Detected {len(groups)} fused groups ({sum(len(g) for g in groups)} layers)")
    return groups


# ---------------------------------------------------------------------------
# Config generation from ILP solution
# ---------------------------------------------------------------------------

_EXPERT_RE = re.compile(r"^(.+\.experts)\.\d+\.(.+)$")


def _collapse_expert_patterns(layer_list: list[str]) -> list[str]:
    """Collapse individual MoE expert layers into wildcard regex patterns."""
    expert_groups: dict[str, set[str]] = defaultdict(set)
    non_expert = []

    for layer in layer_list:
        m = _EXPERT_RE.match(layer)
        if m:
            expert_groups[m.group(1)].add(m.group(2))
        else:
            non_expert.append(f"re:^{re.escape(layer)}$")

    targets = list(non_expert)
    for prefix in sorted(expert_groups):
        suffixes = sorted(expert_groups[prefix])
        escaped = re.escape(prefix)
        if len(suffixes) == 1:
            targets.append(f"re:^{escaped}\\.\\d+\\.{re.escape(suffixes[0])}$")
        else:
            alt = "|".join(re.escape(s) for s in suffixes)
            targets.append(f"re:^{escaped}\\.\\d+\\.({alt})$")

    return targets


def generate_config_groups(
    ilp_solution: Dict[str, str],
    candidate_schemes: Dict[str, QuantizationScheme],
) -> Dict[str, QuantizationScheme]:
    """Convert ILP layer->scheme mapping into config_groups with proper targets and formats."""
    scheme_to_layers: dict[str, list[str]] = defaultdict(list)
    for layer, scheme_name in ilp_solution.items():
        scheme_to_layers[scheme_name].append(layer)

    config_groups = {}
    for idx, (scheme_name, layer_list) in enumerate(sorted(scheme_to_layers.items())):
        base_scheme = candidate_schemes.get(scheme_name)
        if base_scheme is None:
            logger.warning(f"Scheme {scheme_name} not in candidate_schemes, skipping")
            continue

        scheme_dict = base_scheme.model_dump()
        scheme_dict["targets"] = _collapse_expert_patterns(sorted(layer_list))

        if base_scheme.format is None:
            dummy = torch.nn.Linear(64, 64, bias=False)
            initialize_module_for_quantization(dummy, base_scheme, force_zero_point=False)
            scheme_dict["format"] = infer_module_format(type(dummy), base_scheme).value

        group_name = f"group_{idx}_{scheme_name}"
        config_groups[group_name] = QuantizationScheme(**scheme_dict)
        logger.info(f"Config group '{group_name}': {len(layer_list)} layers")

    return config_groups


