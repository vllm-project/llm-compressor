"""
HIGGS: Heuristic ILP-Guided Grouped Scheme Mixed-Precision Quantization

This module implements an ILP-based approach to automatically select
optimal quantization schemes for each layer in a model, minimizing
expected perplexity degradation while respecting hardware constraints.
"""

import os
import time
from typing import List, Optional, Union

import torch
from compressed_tensors.entrypoints.convert import (
    build_inverse_weight_maps,
    convert_checkpoint,
)
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
    load_tensors_from_inverse_weight_map,
)
from loguru import logger

from llmcompressor.transformers.compression.higgs.alpha_heuristic import (
    compute_heuristic_alphas,
)
from llmcompressor.transformers.compression.higgs.config_generator import (
    create_quantization_config,
    generate_config_groups,
)
from llmcompressor.transformers.compression.higgs.fusion_utils import (
    detect_fused_groups,
)
from llmcompressor.transformers.compression.higgs.ilp_solver import (
    solve_ilp_mixed_precision,
)
from llmcompressor.transformers.compression.higgs.mse_collector import (
    HiggsMSECollectorConverter,
)
from llmcompressor.transformers.compression.higgs.mse_utils import (
    compute_layer_mse,
    quantize_weight,
)
from llmcompressor.transformers.compression.higgs.quantization_converter import (
    HiggsQuantizationConverter,
)


__all__ = [
    "ilp_quantize",
    "HiggsMSECollectorConverter",
    "HiggsQuantizationConverter",
    "compute_layer_mse",
    "quantize_weight",
    "solve_ilp_mixed_precision",
    "generate_config_groups",
    "create_quantization_config",
    "compute_heuristic_alphas",
    "detect_fused_groups",
]


def ilp_quantize(
    model_stub: Union[str, os.PathLike],
    save_directory: Union[str, os.PathLike],
    candidate_schemes: List[Union[str, QuantizationScheme]],
    targets: Union[str, List[str]] = "Linear",
    ignore: List[str] = None,
    ilp_solver: str = "pulp",
    enforce_fused_layer_constraints: bool = True,
    target_avg_bitwidth: Optional[float] = None,
    target_avg_act_bitwidth: Optional[float] = None,
    max_workers: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    quantize: bool = True,
) -> QuantizationConfig:
    """
    High-level API for ILP-based mixed-precision quantization.

    Automatically handles two-phase process:
    1. MSE collection and ILP optimization (no checkpoint saved)
    2. Quantization with selected schemes (checkpoint saved)

    Args:
        model_stub: HuggingFace model ID or path to model directory
        save_directory: Where to save quantized model
        candidate_schemes: List of quantization schemes to choose from
        targets: Layer types to quantize (e.g., "Linear")
        ignore: Layers to skip (e.g., ["lm_head"])
        ilp_solver: ILP solver to use ("pulp")
        enforce_fused_layer_constraints: Ensure fused layers get same scheme
        target_avg_bitwidth: Optional constraint on average weight bitwidth
        target_avg_act_bitwidth: Optional constraint on average activation bitwidth
        max_workers: Number of parallel workers for phase 2
        device: Device for quantization (GPU recommended)
        quantize: If False, skip Phase 2 and return config only (for use
            with oneshot or other application methods)

    Returns:
        QuantizationConfig with optimized config_groups

    Example:
        >>> from llmcompressor.transformers.compression.higgs import ilp_quantize
        >>> config = ilp_quantize(
        ...     model_stub="facebook/opt-125m",
        ...     save_directory="./opt-125m-higgs",
        ...     candidate_schemes=["W4A16", "W8A8"],
        ...     targets="Linear",
        ...     ignore=["lm_head"],
        ...     max_workers=4,
        ... )
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ignore is None:
        ignore = ["lm_head"]

    # Warn if using multiple schemes without bitwidth constraint
    if len(candidate_schemes) > 1 and target_avg_bitwidth is None:
        logger.warning(
            "Using multiple candidate schemes without target_avg_bitwidth constraint. "
            "ILP will select the scheme with lowest MSE for each layer (likely highest bitwidth). "
            "Set target_avg_bitwidth to enable true mixed-precision optimization."
        )

    # Phase 1: MSE collection and ILP solving (NO SAVE)
    logger.info("=" * 80)
    logger.info("Phase 1: Collecting MSE data and solving ILP...")
    logger.info("=" * 80)
    phase1_start = time.time()

    # Create MSE collector with heuristic and fusion detection
    collector = HiggsMSECollectorConverter(
        candidate_schemes=candidate_schemes,
        targets=targets,
        ignore=ignore,
        device=device,
        alpha_calculator=compute_heuristic_alphas,
        fusion_detector=detect_fused_groups if enforce_fused_layer_constraints else None,
        ilp_solver=ilp_solver,
        target_avg_bitwidth=target_avg_bitwidth,
        target_avg_act_bitwidth=target_avg_act_bitwidth,
    )

    # Get checkpoint structure
    model_files = get_checkpoint_files(model_stub)
    weight_map = get_weight_map(model_files)
    inverse_weight_maps = build_inverse_weight_maps(
        weight_map=weight_map,
        model_files=model_files,
        converters=[collector],
    )

    # Process each shard WITHOUT saving
    shard_names = [f for f in model_files if f.endswith("safetensors")]
    logger.info(f"Processing {len(shard_names)} shards for MSE collection...")

    mse_start = time.time()
    for shard_name in shard_names:
        if shard_name not in inverse_weight_maps:
            logger.warning(f"Shard {shard_name} not in inverse_weight_maps, skipping")
            continue

        logger.info(f"Processing shard: {shard_name}")
        tensors = load_tensors_from_inverse_weight_map(
            inverse_weight_maps[shard_name], device
        )
        collector.process(tensors)
        # NO SAVE - just accumulating MSE data

    mse_time = time.time() - mse_start
    logger.info(f"MSE collection completed in {mse_time:.2f}s")

    # Solve ILP after all shards processed
    ilp_start = time.time()
    optimal_config = collector.create_config()
    ilp_time = time.time() - ilp_start

    logger.info(
        f"ILP solved in {ilp_time:.2f}s: {len(optimal_config.config_groups)} config groups generated"
    )

    phase1_time = time.time() - phase1_start
    logger.info(f"Phase 1 total time: {phase1_time:.2f}s")

    if not quantize:
        logger.info("quantize=False: returning config without applying quantization")
        return optimal_config

    # Phase 2: Apply quantization with ILP-selected schemes (WITH SAVE)
    logger.info("=" * 80)
    logger.info("Phase 2: Applying ILP-selected quantization schemes...")
    logger.info("=" * 80)
    phase2_start = time.time()

    quantizer = HiggsQuantizationConverter(
        optimal_config=optimal_config,
        targets=targets,
        ignore=ignore,
        device=device,
    )

    convert_checkpoint(
        model_stub=model_stub,  # Re-read original model
        save_directory=save_directory,
        converter=quantizer,
        max_workers=max_workers,
    )

    phase2_time = time.time() - phase2_start
    total_time = time.time() - phase1_start

    logger.info("=" * 80)
    logger.info(f"Quantized model saved to {save_directory}")
    logger.info(f"Phase 2 time: {phase2_time:.2f}s")
    logger.info(f"Total HIGGS time: {total_time:.2f}s (MSE: {mse_time:.2f}s, ILP: {ilp_time:.2f}s, Quantize: {phase2_time:.2f}s)")
    logger.info("=" * 80)

    return optimal_config

