"""
Validate that logically-related layers receive consistent quantization schemes.

Fused layer groups (QKV attention projections, gate/up MLP projections) and MoE
expert replicas should normally share the same quantization configuration.
Mismatches usually indicate a recipe misconfiguration — e.g. a target pattern
that accidentally covers only some projections in a group.

This module warns (rather than errors) because advanced users may intentionally
assign different schemes to related layers.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch
from compressed_tensors.quantization import QuantizationScheme

from llmcompressor.observers.helpers import FUSED_LAYER_NAMES

__all__ = [
    "get_fused_group_mismatches",
    "get_expert_scheme_mismatches",
    "validate_scheme_consistency",
]

logger = logging.getLogger(__name__)


def _scheme_signature(scheme: QuantizationScheme) -> dict:
    """Extract the quantization-relevant fields for comparison and display."""
    sig = {}
    for component in ("weights", "input_activations", "output_activations"):
        args = getattr(scheme, component, None)
        if args is None:
            sig[component] = None
        else:
            sig[component] = {
                "num_bits": args.num_bits,
                "type": args.type,
                "symmetric": args.symmetric,
                "strategy": args.strategy,
                "group_size": args.group_size,
                "dynamic": args.dynamic,
            }
    return sig


def _schemes_match(a: QuantizationScheme, b: QuantizationScheme) -> bool:
    """Compare two schemes on their quantization-relevant fields only."""
    return _scheme_signature(a) == _scheme_signature(b)


def get_fused_group_mismatches(
    model: torch.nn.Module,
) -> list[tuple[str, tuple[str, ...], dict[str, dict]]]:
    """
    Find fused layer groups where members have different quantization schemes.

    Checks groups defined in FUSED_LAYER_NAMES (QKV, gate/up, etc.).

    :param model: model with quantization schemes already applied
    :return: list of (parent_fqn, group_names, {layer_name: scheme_signature})
    """
    mismatches = []

    for parent_name, parent_module in model.named_modules():
        for fusion_group in FUSED_LAYER_NAMES:
            if not all(hasattr(parent_module, name) for name in fusion_group):
                continue

            layers = {}
            for name in fusion_group:
                child = getattr(parent_module, name)
                if child is None:
                    continue
                scheme = getattr(child, "quantization_scheme", None)
                if scheme is not None:
                    layers[name] = scheme

            if len(layers) < 2:
                continue

            schemes = list(layers.values())
            reference = schemes[0]
            if not all(_schemes_match(reference, s) for s in schemes[1:]):
                sigs = {
                    name: _scheme_signature(scheme)
                    for name, scheme in layers.items()
                }
                mismatches.append((parent_name, fusion_group, sigs))

    return mismatches


def get_expert_scheme_mismatches(
    model: torch.nn.Module,
) -> list[tuple[str, str, dict[str, dict]]]:
    """
    Find MoE expert groups where replicas of the same sub-layer have different
    quantization schemes.

    Walks the model looking for ``nn.ModuleList`` containers named ``experts``,
    then groups their children by sub-layer name and checks for consistency.

    :param model: model with quantization schemes already applied
    :return: list of (experts_fqn, sub_layer_name, {expert_idx: scheme_signature})
    """
    mismatches = []

    for parent_name, parent_module in model.named_modules():
        if not (
            isinstance(parent_module, torch.nn.ModuleList)
            and parent_name.endswith("experts")
        ):
            continue

        sublayer_schemes: dict[str, dict[str, QuantizationScheme]] = defaultdict(dict)

        for expert_idx, expert_module in enumerate(parent_module):
            for child_name, child_module in expert_module.named_modules():
                if child_name == "":
                    continue
                scheme = getattr(child_module, "quantization_scheme", None)
                if scheme is not None:
                    sublayer_schemes[child_name][str(expert_idx)] = scheme

        for sublayer_name, expert_schemes in sublayer_schemes.items():
            if len(expert_schemes) < 2:
                continue

            schemes = list(expert_schemes.values())
            reference = schemes[0]
            if not all(_schemes_match(reference, s) for s in schemes[1:]):
                sigs = {
                    idx: _scheme_signature(scheme)
                    for idx, scheme in expert_schemes.items()
                }
                mismatches.append((parent_name, sublayer_name, sigs))

    return mismatches


def _format_fused_mismatches(
    mismatches: list[tuple[str, tuple[str, ...], dict[str, dict]]],
) -> str:
    lines = []
    for parent_fqn, group_names, sigs in mismatches:
        lines.append(f"  Fused group {group_names} under '{parent_fqn}':")
        for layer_name, sig in sigs.items():
            lines.append(f"    {layer_name}: {sig}")
    return "\n".join(lines)


def _format_expert_mismatches(
    mismatches: list[tuple[str, str, dict[str, dict]]],
) -> str:
    lines = []
    for experts_fqn, sublayer_name, sigs in mismatches:
        lines.append(
            f"  Expert sub-layer '{sublayer_name}' under '{experts_fqn}':"
        )
        for idx, sig in sigs.items():
            lines.append(f"    expert {idx}: {sig}")
    return "\n".join(lines)


def validate_scheme_consistency(model: torch.nn.Module) -> None:
    """
    Warn if logically-related layers have inconsistent quantization schemes.

    Checks both fused layer groups (QKV, gate/up) and MoE expert replicas.

    :param model: model with quantization schemes already applied (after
        apply_quantization_config)
    """
    fused = get_fused_group_mismatches(model)
    expert = get_expert_scheme_mismatches(model)

    if not fused and not expert:
        return

    parts = [
        "Inconsistent quantization schemes detected across related layers. "
        "This usually indicates a recipe misconfiguration where target patterns "
        "do not uniformly cover all members of a fused group or all experts."
    ]

    if fused:
        parts.append("\nFused layer group mismatches:\n" + _format_fused_mismatches(fused))
    if expert:
        parts.append("\nMoE expert mismatches:\n" + _format_expert_mismatches(expert))

    logger.warning("\n".join(parts))
