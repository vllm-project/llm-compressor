"""
HIGGS: Heuristic ILP-Guided Grouped Scheme Mixed-Precision Quantization

Converters and high-level API for two-phase ILP-based mixed-precision quantization.
Phase 1 (HiggsMSECollectorConverter): collect per-layer MSE, solve ILP.
Phase 2 (HiggsQuantizationConverter): apply selected schemes with proper fusion.
"""

import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Union

import torch
from compressed_tensors.compressors import compress_module
from compressed_tensors.entrypoints.convert import (
    Converter,
    build_inverse_weight_maps,
)
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    preset_name_to_scheme,
)
from compressed_tensors.utils import match_quantizable_tensors
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
    load_tensors_from_inverse_weight_map,
)
from loguru import logger
from torch.nn import Module

from llmcompressor.entrypoints.model_free.lifecycle import (
    calibrate_weight,
    initialize_quantized_linear,
)
from llmcompressor.entrypoints.model_free.microscale import (
    DEFAULT_FUSED_MAPPINGS,
    get_fused_names,
    is_microscale_scheme,
)
from llmcompressor.entrypoints.model_free.process import split_fused_moe_experts
from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    observe,
    update_qparams,
)
from llmcompressor.observers import FusionHandler
from compressed_tensors.compressors.format import _flatten_formats
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationStatus
from llmcompressor.entrypoints.higgs.utils import (
    compute_heuristic_alphas,
    compute_layer_mse,
    detect_fused_groups,
    generate_config_groups,
)
from llmcompressor.entrypoints.higgs.ilp_solver import (
    solve_ilp_mixed_precision,
)


__all__ = [
    "HiggsMSECollectorConverter",
    "HiggsQuantizationConverter",
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
    ):
        self.targets = targets if isinstance(targets, list) else [targets]
        self.ignore = ignore or ["lm_head"]
        self.device = device or torch.device("cpu")
        self.alpha_calculator = alpha_calculator
        self.fusion_detector = fusion_detector
        self.target_avg_bitwidth = target_avg_bitwidth
        self.target_avg_act_bitwidth = target_avg_act_bitwidth

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
                    key += f"A{scheme.input_activations.num_bits}" if scheme.input_activations else "A16"
                else:
                    key = f"scheme_{len(resolved)}"
                resolved[key] = scheme
            else:
                raise ValueError(f"Invalid scheme type: {type(scheme)}")
        return resolved

    def validate(self, tensors: Dict[str, torch.Tensor]):
        tensors = split_fused_moe_experts(tensors)
        count = sum(1 for _ in match_quantizable_tensors(tensors, self.ignore, self.targets))
        if count == 0:
            logger.warning(f"No quantizable tensors. Targets: {self.targets}, Ignore: {self.ignore}")

    def process(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute per-layer MSE for each candidate scheme. Returns tensors unchanged."""
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

        ilp_solution = solve_ilp_mixed_precision(
            mse_matrix=self.mse_matrix,
            alphas=alphas,
            candidate_schemes=list(self.candidate_schemes.keys()),
            fused_groups=fused_groups,
            target_avg_bitwidth=self.target_avg_bitwidth,
            layer_param_counts=self.layer_sizes if (has_bw or has_act) else None,
            scheme_bitwidths=_bitwidths("weights") if has_bw else None,
            target_avg_act_bitwidth=self.target_avg_act_bitwidth,
            scheme_act_bitwidths=_bitwidths("input_activations") if has_act else None,
        )

        config_groups = generate_config_groups(ilp_solution, self.candidate_schemes)
        formats = {CompressionFormat(s.format) for s in config_groups.values() if s.format}
        self.optimal_config = QuantizationConfig(
            config_groups=config_groups,
            format=_flatten_formats(formats).value,
            quantization_status=QuantizationStatus.COMPRESSED,
            ignore=self.ignore,
        )
        logger.info(f"ILP optimization complete: {len(config_groups)} config groups generated")
        return self.optimal_config

    def get_dependencies(self, weight_name: str) -> set[str]:
        return set()


# ---------------------------------------------------------------------------
# Phase 2: Quantization application
# ---------------------------------------------------------------------------


class HiggsQuantizationConverter(Converter):
    """
    Second-pass converter that applies ILP-selected quantization schemes.

    For microscale schemes (NVFP4A16, NVFP4, MXFP4), fused layer groups
    (q/k/v, gate/up) are processed together with shared global scales.
    """

    def __init__(
        self,
        optimal_config: QuantizationConfig,
        targets: Union[str, List[str]] = "Linear",
        ignore: List[str] = None,
        device: Union[str, torch.device] = None,
    ):
        if optimal_config is None:
            raise ValueError("optimal_config cannot be None")

        self.optimal_config = optimal_config
        self.targets = targets if isinstance(targets, list) else [targets]
        self.ignore = ignore or ["lm_head"]
        self.device = device or torch.device("cpu")

        self._microscale_groups = {}
        self._standard_groups = {}
        for name, scheme in optimal_config.config_groups.items():
            if scheme.weights and is_microscale_scheme(scheme):
                self._microscale_groups[name] = scheme
            else:
                self._standard_groups[name] = scheme

        self._has_microscale = len(self._microscale_groups) > 0

    def validate(self, tensors: Dict[str, torch.Tensor]):
        if not self.optimal_config.config_groups:
            raise ValueError(
                "No config_groups in optimal_config. "
                "Ensure HiggsMSECollectorConverter ran first and solved ILP."
            )

        tensors = split_fused_moe_experts(tensors)
        quantizable_count = 0
        for _ in match_quantizable_tensors(tensors, self.ignore, self.targets):
            quantizable_count += 1

        if quantizable_count == 0:
            logger.warning(
                f"No quantizable tensors found in this shard. "
                f"Targets: {self.targets}, Ignore: {self.ignore}"
            )

    def process(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply ILP-selected quantization scheme to each layer.

        Standard schemes (FP8_DYNAMIC, W4A16): per-layer initialize -> calibrate -> compress.
        Microscale schemes (NVFP4A16, NVFP4): fused layer groups share global scales
        via FusionHandler, matching model_free_ptq behavior.
        """
        tensors = split_fused_moe_experts(tensors)

        logger.info(
            f"Applying ILP-selected quantization to shard with {len(tensors)} tensors"
        )

        layers_quantized = 0
        lookup_time = 0.0
        quant_time = 0.0

        # --- Standard (non-microscale) groups ---
        for config_group_name, scheme in self._standard_groups.items():
            logger.debug(f"Processing standard config group: {config_group_name}")

            lookup_start = time.time()
            matches = list(match_quantizable_tensors(
                tensors, self.ignore, scheme.targets
            ))
            lookup_time += time.time() - lookup_start

            for module_name, tensor_name in matches:
                weight = tensors[tensor_name]
                logger.debug(f"  Quantizing {module_name} with scheme {config_group_name}")

                try:
                    quant_start = time.time()
                    module = initialize_quantized_linear(weight, scheme, self.device)
                    calibrate_weight(module)
                    compress_module(module)

                    del tensors[tensor_name]
                    prefix = module_name + "."
                    for key, value in module.state_dict(prefix=prefix).items():
                        tensors[key] = value.to("cpu")

                    quant_time += time.time() - quant_start
                    layers_quantized += 1

                except Exception as e:
                    logger.error(
                        f"Failed to quantize {module_name} with {config_group_name}: {e}"
                    )
                    continue

        # --- Microscale groups (need fused global scale handling) ---
        if self._has_microscale:
            fused_sets, _ = get_fused_names(list(tensors.keys()))
            fused_name_to_fused_index: Dict[str, int] = {
                name: index
                for index, matched_set in enumerate(fused_sets)
                for name in matched_set.values()
                if name is not None
            }
            fused_modules: Dict[int, Dict[str, Module]] = defaultdict(dict)

            for config_group_name, scheme in self._microscale_groups.items():
                logger.debug(
                    f"Processing microscale config group: {config_group_name}"
                )

                lookup_start = time.time()
                matches = list(match_quantizable_tensors(
                    tensors, self.ignore, scheme.targets
                ))
                lookup_time += time.time() - lookup_start

                for module_name, tensor_name in matches:
                    weight = tensors[tensor_name]

                    try:
                        quant_start = time.time()
                        module = initialize_quantized_linear(
                            weight, scheme, self.device
                        )

                        # Fused layers: defer calibration for shared global scale
                        if tensor_name in fused_name_to_fused_index:
                            fused_index = fused_name_to_fused_index[tensor_name]
                            fused_modules[fused_index][tensor_name] = module
                            initialize_observer(module, "weight")
                            apply_calibration_status(module)
                            quant_time += time.time() - quant_start
                            continue

                        # Non-fused microscale: standard path
                        calibrate_weight(module)
                        compress_module(module)

                        del tensors[tensor_name]
                        prefix = module_name + "."
                        for key, value in module.state_dict(prefix=prefix).items():
                            tensors[key] = value.to("cpu")

                        quant_time += time.time() - quant_start
                        layers_quantized += 1

                    except Exception as e:
                        logger.error(
                            f"Failed to quantize {module_name} with "
                            f"{config_group_name}: {e}"
                        )
                        continue

            # Compress fused modules with shared global scale
            for named_modules in fused_modules.values():
                quant_start = time.time()

                FusionHandler.fuse(
                    [
                        (mod.weight_observer, mod)
                        for mod in named_modules.values()
                    ]
                )
                observe(named_modules.values(), base_name="weight")
                update_qparams(named_modules.values(), base_name="weight")

                for name, module in named_modules.items():
                    freeze_module_quantization(module)
                    compress_module(module)

                    del tensors[name]
                    module_name, _ = name.rsplit(".", 1)
                    prefix = module_name + "."
                    for key, value in module.state_dict(prefix=prefix).items():
                        tensors[key] = value.to("cpu")

                    layers_quantized += 1

                quant_time += time.time() - quant_start

        logger.info(
            f"Quantized {layers_quantized} layers in this shard "
            f"(lookup: {lookup_time:.2f}s, quant: {quant_time:.2f}s)"
        )

        return tensors

    def create_config(self) -> QuantizationConfig:
        return self.optimal_config

    def update_config(
        self, config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        return self.optimal_config

    def get_dependencies(self, weight_name: str) -> set[str]:
        """Return fused partner dependencies for microscale weights."""
        if not self._has_microscale:
            return set()

        deps = set()
        for primary_pattern, partner_templates in DEFAULT_FUSED_MAPPINGS.items():
            match = re.match(primary_pattern, weight_name)
            if match is None:
                continue
            for partner_template in partner_templates:
                partner_name = partner_template.format(**match.groupdict())
                deps.add(partner_name)
        return deps


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
) -> QuantizationConfig:
    """
    Compute optimal mixed-precision config via ILP on per-layer MSE.

    Reads model weights (model-free, no GPU model load), evaluates each
    candidate scheme's MSE on every layer, and solves an ILP to minimize
    weighted MSE under a bitwidth budget.

    The returned QuantizationConfig can be applied via oneshot() or
    convert_checkpoint().

    Args:
        model_stub: HuggingFace model ID or path to model directory
        candidate_schemes: List of quantization schemes to choose from
        targets: Layer types to quantize (e.g., "Linear")
        ignore: Layers to skip (e.g., ["lm_head"])
        enforce_fused_layer_constraints: Ensure fused layers get same scheme
        target_avg_bitwidth: Optional constraint on average weight bitwidth
        target_avg_act_bitwidth: Optional constraint on average activation bitwidth
        device: Device for MSE computation (GPU recommended)

    Returns:
        QuantizationConfig with optimized config_groups
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ignore is None:
        ignore = ["lm_head"]

    if len(candidate_schemes) > 1 and target_avg_bitwidth is None:
        logger.warning(
            "Using multiple candidate schemes without target_avg_bitwidth constraint. "
            "ILP will select the scheme with lowest MSE for each layer (likely highest bitwidth). "
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
        fusion_detector=detect_fused_groups if enforce_fused_layer_constraints else None,
        target_avg_bitwidth=target_avg_bitwidth,
        target_avg_act_bitwidth=target_avg_act_bitwidth,
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
