"""
Heuristic calculation of layer importance (alpha) values.

Based on the provided heuristic.py, this module computes alpha values
for each layer using a formula that considers layer size, depth, and type.
"""

import re
from typing import Dict, List

import numpy as np
from loguru import logger


__all__ = ["compute_heuristic_alphas"]


def compute_heuristic_alphas(
    layer_names: List[str],
    layer_param_counts: Dict[str, int],
) -> Dict[str, float]:
    """
    Compute heuristic alpha (importance weight) for each layer.

    Formula: alpha = log(size + 1) * (1 + depth * 0.05) * type_multiplier

    Where:
    - size: number of parameters in layer
    - depth: layer index in network (0-indexed)
    - type_multiplier:
        * embedding: 1.5 (most important)
        * attention: 1.2
        * mlp: 0.9
        * default: 1.0

    Args:
        layer_names: List of layer names (e.g., "model.layers.0.self_attn.q_proj")
        layer_param_counts: Dict mapping layer_name -> parameter count

    Returns:
        Dict mapping layer_name -> alpha value

    Example:
        >>> layer_names = ["model.layers.0.self_attn.q_proj", "model.layers.5.mlp.gate_proj"]
        >>> param_counts = {"model.layers.0.self_attn.q_proj": 1000, "model.layers.5.mlp.gate_proj": 2000}
        >>> alphas = compute_heuristic_alphas(layer_names, param_counts)
        >>> # alphas["model.layers.0.self_attn.q_proj"] ≈ log(1001) * 1.0 * 1.2 = 8.3
        >>> # alphas["model.layers.5.mlp.gate_proj"] ≈ log(2001) * 1.25 * 0.9 = 8.6
    """
    logger.info(f"Computing heuristic alphas for {len(layer_names)} layers...")

    heuristic_alphas: Dict[str, float] = {}

    for layer_name in layer_names:
        layer_size = layer_param_counts.get(layer_name, 0)

        # Extract metadata from layer name
        meta = _extract_layer_metadata(layer_name, layer_size)

        if layer_size == 0:
            logger.warning(
                f"No param count for {layer_name}, using default alpha=1.0"
            )
            heuristic_alphas[layer_name] = 1.0
            continue

        # Base alpha: log of parameter count
        base_alpha = np.log(layer_size + 1)

        # Depth factor: deeper layers weighted slightly higher
        depth_factor = 1 + meta["depth"] * 0.05

        # Type multiplier: different layer types have different importance
        type_multiplier = 1.0
        if meta["type"] == "attention":
            type_multiplier = 1.2
        elif meta["type"] == "embedding":
            type_multiplier = 1.5
        elif meta["type"] == "mlp":
            type_multiplier = 0.9

        # Combined alpha value
        alpha_val = float(base_alpha * depth_factor * type_multiplier)
        heuristic_alphas[layer_name] = alpha_val

    logger.info(f"Heuristic alphas computed for {len(heuristic_alphas)} layers")

    # Log statistics
    if heuristic_alphas:
        alpha_values = list(heuristic_alphas.values())
        logger.info(
            f"  Alpha range: [{min(alpha_values):.2f}, {max(alpha_values):.2f}]"
        )
        logger.info(f"  Alpha mean: {np.mean(alpha_values):.2f}")

    return heuristic_alphas


def _extract_layer_metadata(layer_name: str, size: int) -> Dict[str, object]:
    """
    Extract depth and type from layer name.

    Examples:
    - "model.layers.0.self_attn.q_proj" -> depth=0, type="attention"
    - "model.layers.15.mlp.gate_proj" -> depth=15, type="mlp"
    - "model.embed_tokens" -> depth=0, type="embedding"

    Args:
        layer_name: Full layer name
        size: Number of parameters (for logging)

    Returns:
        Dict with keys: "name", "size", "depth", "type"
    """
    meta = {
        "name": layer_name,
        "size": size,
        "depth": 0,
        "type": "default",
    }

    # Extract depth (layer index) from common patterns
    # Pattern: "layers.{N}." or "layer.{N}." or "h.{N}."
    depth_patterns = [
        r"layers\.(\d+)\.",
        r"layer\.(\d+)\.",
        r"h\.(\d+)\.",
        r"blocks\.(\d+)\.",
    ]

    for pattern in depth_patterns:
        match = re.search(pattern, layer_name)
        if match:
            meta["depth"] = int(match.group(1))
            break

    # Extract type from layer name
    layer_lower = layer_name.lower()

    if "embed" in layer_lower:
        meta["type"] = "embedding"
    elif any(
        attn_keyword in layer_lower
        for attn_keyword in ["attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj"]
    ):
        meta["type"] = "attention"
    elif any(
        mlp_keyword in layer_lower
        for mlp_keyword in ["mlp", "ffn", "fc", "gate_proj", "up_proj", "down_proj"]
    ):
        meta["type"] = "mlp"

    return meta
