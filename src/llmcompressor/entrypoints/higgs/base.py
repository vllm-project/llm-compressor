"""
HIGGS: Heuristic ILP-Guided Grouped Scheme Mixed-Precision Quantization

MSE collection and high-level API for ILP-based mixed-precision quantization.
"""

import os
import time
from typing import Dict, List, Optional, Union

import torch
from compressed_tensors.compressors.format import _flatten_formats
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert import (
    Converter,
    build_inverse_weight_maps,
)
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    preset_name_to_scheme,
)
from compressed_tensors.utils import match_quantizable_tensors
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
    load_tensors_from_inverse_weight_map,
)
from loguru import logger

from llmcompressor.entrypoints.higgs.ilp_solver import (
    solve_ilp_mixed_precision,
)
from llmcompressor.entrypoints.higgs.utils import (
    UNQUANTIZED_SCHEME,
    compute_heuristic_alphas,
    compute_layer_mse,
    detect_fused_groups,
    generate_config_groups,
)
from llmcompressor.entrypoints.model_free.converter import split_fused_moe_experts

__all__ = [
    "HiggsMSECollectorConverter",
    "get_higgs_config",
]


# ---------------------------------------------------------------------------
# Phase 1: MSE collection and ILP solving
# ---------------------------------------------------------------------------


class HiggsMSECollectorConverter(Converter):
    """
    First-pass converter that collects per-layer MSE data, solves the ILP,
    and produces an optimal QuantizationConfig. Does NOT modify tensors.
    """

    def __init__(
        self,
        candidate_schemes: List[Union[str, QuantizationScheme]],
        targets: Union[str, List[str]] = "Linear",
        ignore: List[str] = None,
        device: Union[str, torch.device] = None,
        alpha_calculator: callable = None,
        fusion_detector: callable = None,
        target_avg_bitwidth: Optional[float] = None,
        target_avg_act_bitwidth: Optional[float] = None,
        allow_unquantized: bool = True,
    ):
        self.targets = targets if isinstance(targets, list) else [targets]
        self.ignore = ignore or ["lm_head"]
        self.device = device or torch.device("cpu")
        self.alpha_calculator = alpha_calculator
        self.fusion_detector = fusion_detector
        self.target_avg_bitwidth = target_avg_bitwidth
        self.target_avg_act_bitwidth = target_avg_act_bitwidth
        self.allow_unquantized = allow_unquantized

        self.candidate_schemes = self._resolve_schemes(candidate_schemes)

        self.mse_matrix: Dict[str, Dict[str, float]] = {}
        self.layer_sizes: Dict[str, int] = {}
        self.optimal_config: Optional[QuantizationConfig] = None

    def _resolve_schemes(
        self, schemes: List[Union[str, QuantizationScheme]]
    ) -> Dict[str, QuantizationScheme]:
        resolved = {}
        for scheme in schemes:
            if isinstance(scheme, str):
                resolved[scheme] = preset_name_to_scheme(scheme, targets=self.targets)
            elif isinstance(scheme, QuantizationScheme):
                if scheme.weights:
                    key = f"W{scheme.weights.num_bits}"
                    key += (
                        f"A{scheme.input_activations.num_bits}"
                        if scheme.input_activations
                        else "A16"
                    )
                else:
                    key = f"scheme_{len(resolved)}"
                resolved[key] = scheme
            else:
                raise ValueError(f"Invalid scheme type: {type(scheme)}")
        return resolved

    def validate(self, tensors: Dict[str, torch.Tensor]):
        """Check that there are quantizable tensors in this shard. Note: this is currently
           never called by the HIGGS workflow"""
        tensors = split_fused_moe_experts(tensors)
        count = sum(
            1 for _ in match_quantizable_tensors(tensors, self.ignore, self.targets)
        )
        if count == 0:
            logger.warning(
                f"No quantizable tensors. Targets: {self.targets}, "
                f"Ignore: {self.ignore}"
            )

    def process(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute MSE for each candidate scheme; return tensors unchanged."""
        tensors = split_fused_moe_experts(tensors)
        logger.info(f"Collecting MSE data from shard with {len(tensors)} tensors")

        layers_processed = 0
        for module_name, tensor_name in match_quantizable_tensors(
            tensors, self.ignore, self.targets
        ):
            self.layer_sizes[module_name] = tensors[tensor_name].numel()
            if module_name not in self.mse_matrix:
                self.mse_matrix[module_name] = {}

            for scheme_name, scheme in self.candidate_schemes.items():
                if scheme_name not in self.mse_matrix[module_name]:
                    self.mse_matrix[module_name][scheme_name] = compute_layer_mse(
                        tensors[tensor_name], scheme, self.device
                    )
            if self.allow_unquantized:
                self.mse_matrix[module_name][UNQUANTIZED_SCHEME] = 0.0
            layers_processed += 1

        logger.info(f"Processed {layers_processed} layers in this shard")
        return tensors

    def create_config(self) -> QuantizationConfig:
        """Solve ILP after all shards processed and return optimized config."""
        if self.optimal_config is not None:
            return self.optimal_config

        logger.info(
            f"Solving ILP for {len(self.mse_matrix)} layers "
            f"with {len(self.candidate_schemes)} candidate schemes"
        )

        alphas = (
            self.alpha_calculator(list(self.mse_matrix.keys()), self.layer_sizes)
            if self.alpha_calculator
            else {layer: 1.0 for layer in self.mse_matrix}
        )

        fused_groups = (
            self.fusion_detector(list(self.mse_matrix.keys()))
            if self.fusion_detector
            else []
        )

        def _bitwidths(attr):
            return {
                name: float(getattr(s, attr).num_bits) if getattr(s, attr) else 16.0
                for name, s in self.candidate_schemes.items()
            }

        has_bw = self.target_avg_bitwidth is not None
        has_act = self.target_avg_act_bitwidth is not None

        ilp_candidate_schemes = list(self.candidate_schemes.keys())
        if self.allow_unquantized:
            ilp_candidate_schemes.append(UNQUANTIZED_SCHEME)

        weight_bitwidths = _bitwidths("weights") if has_bw else None
        activation_bitwidths = _bitwidths("input_activations") if has_act else None
        if self.allow_unquantized:
            if weight_bitwidths is not None:
                weight_bitwidths[UNQUANTIZED_SCHEME] = 16.0
            if activation_bitwidths is not None:
                activation_bitwidths[UNQUANTIZED_SCHEME] = 16.0

        ilp_solution = solve_ilp_mixed_precision(
            mse_matrix=self.mse_matrix,
            alphas=alphas,
            candidate_schemes=ilp_candidate_schemes,
            fused_groups=fused_groups,
            target_avg_bitwidth=self.target_avg_bitwidth,
            layer_param_counts=self.layer_sizes if (has_bw or has_act) else None,
            scheme_bitwidths=weight_bitwidths,
            target_avg_act_bitwidth=self.target_avg_act_bitwidth,
            scheme_act_bitwidths=activation_bitwidths,
        )

        config_groups = generate_config_groups(ilp_solution, self.candidate_schemes)
        formats = {
            CompressionFormat(s.format) for s in config_groups.values() if s.format
        }
        self.optimal_config = QuantizationConfig(
            config_groups=config_groups,
            format=_flatten_formats(formats).value,
            quantization_status=QuantizationStatus.COMPRESSED,
            ignore=self.ignore,
        )
        logger.info(
            f"ILP optimization complete: {len(config_groups)} config groups generated"
        )
        return self.optimal_config

    def get_dependencies(self, weight_name: str) -> set[str]:
        return set()


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def get_higgs_config(
    model_stub: Union[str, os.PathLike],
    candidate_schemes: List[Union[str, QuantizationScheme]],
    targets: Union[str, List[str]] = "Linear",
    ignore: List[str] = None,
    enforce_fused_layer_constraints: bool = True,
    target_avg_bitwidth: Optional[float] = None,
    target_avg_act_bitwidth: Optional[float] = None,
    device: Optional[Union[str, torch.device]] = None,
    allow_unquantized: bool = True,
) -> QuantizationConfig:
    """
    Compute optimal mixed-precision config via ILP on per-layer MSE.

    Reads model weights (model-free, no GPU model load), evaluates each
    candidate scheme's MSE on every layer, and solves an ILP to minimize
    weighted MSE under a bitwidth budget.

    The returned QuantizationConfig can be applied via oneshot() or
    model_free_ptq().

    Args:
        model_stub: HuggingFace model ID or path to model directory
        candidate_schemes: List of quantization schemes to choose from
        targets: Layer types to quantize (e.g., "Linear")
        ignore: Layers to skip (e.g., ["lm_head"])
        enforce_fused_layer_constraints: Ensure fused layers get same scheme
        target_avg_bitwidth: Optional constraint on average weight bitwidth
        target_avg_act_bitwidth: Optional constraint on average activation bitwidth
        device: Device for MSE computation (GPU recommended)
        allow_unquantized: Allow the ILP to leave layers unquantized at 16 bits

    Returns:
        QuantizationConfig with optimized config_groups
    """
    if device is None:
        device = torch.device("cuda" if torch.accelerator.is_available() else "cpu")

    if ignore is None:
        ignore = ["lm_head"]

    if (
        len(candidate_schemes) + int(allow_unquantized) > 1
        and target_avg_bitwidth is None
    ):
        logger.warning(
            "Using multiple candidate schemes without target_avg_bitwidth constraint. "
            "ILP will select the scheme with lowest MSE for each layer. "
            "Set target_avg_bitwidth to enable true mixed-precision optimization."
        )

    logger.info("=" * 80)
    logger.info("HIGGS: Collecting MSE data and solving ILP...")
    logger.info("=" * 80)
    start = time.time()

    collector = HiggsMSECollectorConverter(
        candidate_schemes=candidate_schemes,
        targets=targets,
        ignore=ignore,
        device=device,
        alpha_calculator=compute_heuristic_alphas,
        fusion_detector=detect_fused_groups
        if enforce_fused_layer_constraints
        else None,
        target_avg_bitwidth=target_avg_bitwidth,
        target_avg_act_bitwidth=target_avg_act_bitwidth,
        allow_unquantized=allow_unquantized,
    )

    model_files = get_checkpoint_files(model_stub)
    weight_map = get_weight_map(model_files)
    inverse_weight_maps = build_inverse_weight_maps(
        weight_map=weight_map,
        model_files=model_files,
        converters=[collector],
    )

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

    mse_time = time.time() - mse_start
    logger.info(f"MSE collection completed in {mse_time:.2f}s")

    ilp_start = time.time()
    optimal_config = collector.create_config()
    ilp_time = time.time() - ilp_start

    total_time = time.time() - start
    logger.info(
        f"HIGGS complete in {total_time:.2f}s "
        f"(MSE: {mse_time:.2f}s, ILP: {ilp_time:.2f}s): "
        f"{len(optimal_config.config_groups)} config groups"
    )

    return optimal_config
