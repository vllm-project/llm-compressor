"""
Generate QuantizationConfig with config_groups from ILP solution.

This module converts the ILP solver's layer-to-scheme mapping into
a QuantizationConfig object with properly structured config_groups.
"""

import re
from typing import Dict

from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from loguru import logger


__all__ = ["generate_config_groups"]

_EXPERT_RE = re.compile(r"^(.+\.experts)\.\d+\.(.+)$")


def _collapse_expert_patterns(layer_list: list[str]) -> list[str]:
    """
    Collapse individual MoE expert layer names into wildcard regex patterns.

    Instead of 384 separate patterns per MoE layer (128 experts × 3 weight types),
    generates compact patterns like:
        re:^model\\.layers\\.0\\.mlp\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$

    Non-expert layers get individual exact-match patterns as before.
    """
    expert_groups: dict[str, set[str]] = {}
    non_expert = []

    for layer in layer_list:
        m = _EXPERT_RE.match(layer)
        if m:
            prefix, suffix = m.group(1), m.group(2)
            if prefix not in expert_groups:
                expert_groups[prefix] = set()
            expert_groups[prefix].add(suffix)
        else:
            non_expert.append(f"re:^{re.escape(layer)}$")

    targets = list(non_expert)
    for prefix in sorted(expert_groups):
        suffixes = sorted(expert_groups[prefix])
        escaped_prefix = re.escape(prefix)
        if len(suffixes) == 1:
            targets.append(f"re:^{escaped_prefix}\\.\\d+\\.{re.escape(suffixes[0])}$")
        else:
            suffix_alt = "|".join(re.escape(s) for s in suffixes)
            targets.append(f"re:^{escaped_prefix}\\.\\d+\\.({suffix_alt})$")

    return targets


def generate_config_groups(
    ilp_solution: Dict[str, str],
    candidate_schemes: Dict[str, QuantizationScheme],
) -> Dict[str, QuantizationScheme]:
    """
    Convert ILP solution to config_groups format for QuantizationConfig.

    Groups layers by their assigned scheme and creates a config_group
    for each unique scheme, with targets listing all layers using that scheme.

    Args:
        ilp_solution: Dict mapping layer_name -> scheme_name (from ILP solver)
        candidate_schemes: Dict mapping scheme_name -> QuantizationScheme object

    Returns:
        Dict mapping config_group_name -> QuantizationScheme with targets set

    Example:
        >>> ilp_solution = {
        ...     "model.layers.0.q_proj": "W4A16",
        ...     "model.layers.0.k_proj": "W4A16",
        ...     "model.layers.0.mlp.gate_proj": "W8A8",
        ... }
        >>> candidate_schemes = {
        ...     "W4A16": QuantizationScheme(...),
        ...     "W8A8": QuantizationScheme(...),
        ... }
        >>> config_groups = generate_config_groups(ilp_solution, candidate_schemes)
        >>> # Returns:
        >>> # {
        >>> #   "group_0_W4A16": QuantizationScheme(targets=["re:^model\\.layers\\.0\\.q_proj$", ...]),
        >>> #   "group_1_W8A8": QuantizationScheme(targets=["re:^model\\.layers\\.0\\.mlp\\.gate_proj$"]),
        >>> # }
    """
    # Group layers by their assigned scheme
    scheme_to_layers = {}
    for layer, scheme_name in ilp_solution.items():
        if scheme_name not in scheme_to_layers:
            scheme_to_layers[scheme_name] = []
        scheme_to_layers[scheme_name].append(layer)

    # Create config_groups
    config_groups = {}
    for idx, (scheme_name, layer_list) in enumerate(sorted(scheme_to_layers.items())):
        if scheme_name not in candidate_schemes:
            logger.warning(
                f"Scheme {scheme_name} not found in candidate_schemes, skipping"
            )
            continue

        # Get the base scheme
        base_scheme = candidate_schemes[scheme_name]

        # Create a copy and update targets
        # Collapse MoE expert layers into wildcard patterns to avoid
        # O(tensors × patterns) regex explosion in match_quantizable_tensors
        layer_targets = _collapse_expert_patterns(sorted(layer_list))

        # Create new scheme with updated targets
        scheme_dict = base_scheme.model_dump()
        scheme_dict["targets"] = layer_targets

        # Recreate QuantizationScheme with new targets
        config_scheme = QuantizationScheme(**scheme_dict)

        # Add to config_groups
        group_name = f"group_{idx}_{scheme_name}"
        config_groups[group_name] = config_scheme

        logger.info(
            f"Config group '{group_name}': {len(layer_list)} layers with scheme {scheme_name}"
        )

    return config_groups


def create_quantization_config(
    config_groups: Dict[str, QuantizationScheme],
    ignore: list = None,
) -> QuantizationConfig:
    """
    Create a QuantizationConfig from config_groups.

    Args:
        config_groups: Dict of config_group_name -> QuantizationScheme
        ignore: Optional list of layer patterns to ignore

    Returns:
        QuantizationConfig ready to use with converters
    """
    from compressed_tensors.quantization import QuantizationStatus

    # Determine format based on schemes
    formats = set()
    for scheme in config_groups.values():
        if scheme.format:
            formats.add(scheme.format)

    # Use mixed-precision format if multiple schemes
    if len(config_groups) > 1:
        format_str = "pack-quantized"
    elif formats:
        format_str = next(iter(formats))
    else:
        format_str = "pack-quantized"

    return QuantizationConfig(
        config_groups=config_groups,
        format=format_str,
        quantization_status=QuantizationStatus.COMPRESSED,
        ignore=ignore or [],
    )
