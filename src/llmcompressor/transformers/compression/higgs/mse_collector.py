"""
MSE Collector Converter - First pass of ILP mixed-precision quantization.

This converter collects MSE statistics for each layer under each candidate
quantization scheme, then solves an ILP to determine the optimal scheme
assignment.
"""

from typing import Dict, List, Optional, Union

import torch
from compressed_tensors.entrypoints.convert import Converter
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    preset_name_to_scheme,
)
from compressed_tensors.utils import match_quantizable_tensors
from loguru import logger

from llmcompressor.entrypoints.model_free.process import split_fused_moe_experts
from llmcompressor.transformers.compression.higgs.config_generator import (
    create_quantization_config,
    generate_config_groups,
)
from llmcompressor.transformers.compression.higgs.ilp_solver import (
    solve_ilp_mixed_precision,
)
from llmcompressor.transformers.compression.higgs.mse_utils import compute_layer_mse


__all__ = ["HiggsMSECollectorConverter"]


class HiggsMSECollectorConverter(Converter):
    """
    First-pass converter that collects MSE data for ILP optimization.

    This converter processes tensors WITHOUT modifying them. It:
    1. Computes MSE for each layer under each candidate quantization scheme
    2. Accumulates MSE data across all shards
    3. In create_config(), solves ILP and generates optimal QuantizationConfig

    Args:
        candidate_schemes: List of scheme names or QuantizationScheme objects
        targets: Layer types to quantize (e.g., "Linear")
        ignore: Layer patterns to skip
        device: Device for MSE computation
        alpha_calculator: Optional callable that takes (layer_names, layer_sizes)
                         and returns Dict[str, float] of alpha values
        fusion_detector: Optional callable that takes layer_names and returns
                        List[List[str]] of fused groups
        ilp_solver: ILP solver name ("pulp", "ortools", "cvxpy")
        target_avg_bitwidth: Optional constraint on average bitwidth
    """

    def __init__(
        self,
        candidate_schemes: List[Union[str, QuantizationScheme]],
        targets: Union[str, List[str]] = "Linear",
        ignore: List[str] = None,
        device: Union[str, torch.device] = None,
        alpha_calculator: callable = None,
        fusion_detector: callable = None,
        ilp_solver: str = "pulp",
        target_avg_bitwidth: Optional[float] = None,
        target_avg_act_bitwidth: Optional[float] = None,
    ):
        # Set targets first so _resolve_schemes can use it
        self.targets = targets if isinstance(targets, list) else [targets]
        self.ignore = ignore or ["lm_head"]
        self.device = device or torch.device("cpu")
        self.alpha_calculator = alpha_calculator
        self.fusion_detector = fusion_detector
        self.ilp_solver = ilp_solver
        self.target_avg_bitwidth = target_avg_bitwidth
        self.target_avg_act_bitwidth = target_avg_act_bitwidth

        # Resolve schemes after setting targets
        self.candidate_schemes = self._resolve_schemes(candidate_schemes)

        # State accumulated across shards
        self.mse_matrix: Dict[str, Dict[str, float]] = {}
        self.layer_sizes: Dict[str, int] = {}
        self.optimal_config: Optional[QuantizationConfig] = None

    def _resolve_schemes(
        self, schemes: List[Union[str, QuantizationScheme]]
    ) -> Dict[str, QuantizationScheme]:
        """Convert scheme names to QuantizationScheme objects."""
        resolved = {}
        for scheme in schemes:
            if isinstance(scheme, str):
                # Preset scheme name like "W4A16"
                scheme_obj = preset_name_to_scheme(scheme, targets=self.targets)
                resolved[scheme] = scheme_obj
            elif isinstance(scheme, QuantizationScheme):
                # Use a key from the scheme (e.g., based on bitwidth)
                if scheme.weights:
                    key = f"W{scheme.weights.num_bits}"
                    if scheme.input_activations:
                        key += f"A{scheme.input_activations.num_bits}"
                    else:
                        key += "A16"
                else:
                    key = f"scheme_{len(resolved)}"
                resolved[key] = scheme
            else:
                raise ValueError(f"Invalid scheme type: {type(scheme)}")

        return resolved

    def validate(self, tensors: Dict[str, torch.Tensor]):
        """Validate that tensors can be quantized."""
        tensors = split_fused_moe_experts(tensors)
        # Basic validation - check that we have some quantizable tensors
        quantizable_count = 0
        for module_name, tensor_name in match_quantizable_tensors(
            tensors, self.ignore, self.targets
        ):
            quantizable_count += 1

        if quantizable_count == 0:
            logger.warning(
                f"No quantizable tensors found in this shard. "
                f"Targets: {self.targets}, Ignore: {self.ignore}"
            )

    def process(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute MSE for each layer under each candidate scheme.
        Accumulate in state, return tensors UNCHANGED.
        """
        # Split fused 3D MoE expert tensors into individual 2D expert tensors
        tensors = split_fused_moe_experts(tensors)

        logger.info(f"Collecting MSE data from shard with {len(tensors)} tensors")

        layers_processed = 0
        for module_name, tensor_name in match_quantizable_tensors(
            tensors, self.ignore, self.targets
        ):
            weight = tensors[tensor_name]

            # Store layer size (parameter count)
            self.layer_sizes[module_name] = weight.numel()

            # Initialize MSE dict for this layer
            if module_name not in self.mse_matrix:
                self.mse_matrix[module_name] = {}

            # Compute MSE for each candidate scheme
            for scheme_name, scheme in self.candidate_schemes.items():
                if scheme_name not in self.mse_matrix[module_name]:
                    mse = compute_layer_mse(weight, scheme, self.device)
                    self.mse_matrix[module_name][scheme_name] = mse

                    logger.debug(
                        f"  {module_name} + {scheme_name}: MSE = {mse:.6f}"
                    )

            layers_processed += 1

        logger.info(f"Processed {layers_processed} layers in this shard")

        # Return tensors UNCHANGED - no quantization yet
        return tensors

    def create_config(self) -> QuantizationConfig:
        """
        Called after all shards processed. Solve ILP and return optimized config.
        """
        if self.optimal_config is not None:
            # Already solved
            return self.optimal_config

        logger.info(
            f"Solving ILP for {len(self.mse_matrix)} layers "
            f"with {len(self.candidate_schemes)} candidate schemes"
        )

        # Compute alpha values (layer importance weights)
        if self.alpha_calculator:
            alphas = self.alpha_calculator(
                list(self.mse_matrix.keys()),
                self.layer_sizes,
            )
        else:
            # Default: all layers have equal importance
            alphas = {layer: 1.0 for layer in self.mse_matrix}

        # Detect fused layer groups
        if self.fusion_detector:
            fused_groups = self.fusion_detector(list(self.mse_matrix.keys()))
        else:
            fused_groups = []

        # Prepare bitwidth info if needed
        needs_param_counts = (
            self.target_avg_bitwidth is not None
            or self.target_avg_act_bitwidth is not None
        )

        scheme_bitwidths = None
        if self.target_avg_bitwidth is not None:
            scheme_bitwidths = {}
            for name, scheme in self.candidate_schemes.items():
                if scheme.weights:
                    scheme_bitwidths[name] = float(scheme.weights.num_bits)
                else:
                    scheme_bitwidths[name] = 16.0

        scheme_act_bitwidths = None
        if self.target_avg_act_bitwidth is not None:
            scheme_act_bitwidths = {}
            for name, scheme in self.candidate_schemes.items():
                if scheme.input_activations:
                    scheme_act_bitwidths[name] = float(
                        scheme.input_activations.num_bits
                    )
                else:
                    scheme_act_bitwidths[name] = 16.0

        # Solve ILP
        ilp_solution = solve_ilp_mixed_precision(
            mse_matrix=self.mse_matrix,
            alphas=alphas,
            candidate_schemes=list(self.candidate_schemes.keys()),
            fused_groups=fused_groups,
            target_avg_bitwidth=self.target_avg_bitwidth,
            layer_param_counts=self.layer_sizes if needs_param_counts else None,
            scheme_bitwidths=scheme_bitwidths,
            target_avg_act_bitwidth=self.target_avg_act_bitwidth,
            scheme_act_bitwidths=scheme_act_bitwidths,
        )

        # Generate config_groups from ILP solution
        config_groups = generate_config_groups(ilp_solution, self.candidate_schemes)

        # Create QuantizationConfig
        self.optimal_config = create_quantization_config(
            config_groups,
            ignore=self.ignore,
        )

        logger.info(
            f"ILP optimization complete: {len(config_groups)} config groups generated"
        )

        return self.optimal_config

    def get_dependencies(self, weight_name: str) -> set[str]:
        """No dependencies for MSE collection."""
        return set()

    def get_optimal_config(self) -> QuantizationConfig:
        """
        Get the optimal QuantizationConfig after processing all shards.

        Returns:
            QuantizationConfig with optimized config_groups

        Raises:
            RuntimeError: If create_config() hasn't been called yet
        """
        if self.optimal_config is None:
            raise RuntimeError(
                "Optimal config not available. Call create_config() first."
            )
        return self.optimal_config
