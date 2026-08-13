"""
Quantization Application Converter - Second pass of ILP mixed-precision quantization.

This converter applies the ILP-selected quantization schemes to each layer
using llmcompressor's quantization infrastructure.
"""

import time
from typing import Dict, List, Union

import torch
from compressed_tensors.compressors import compress_module
from compressed_tensors.entrypoints.convert import Converter
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.utils import match_quantizable_tensors
from loguru import logger

from llmcompressor.entrypoints.model_free.process import split_fused_moe_experts
from llmcompressor.entrypoints.model_free.lifecycle import (
    calibrate_weight,
    initialize_quantized_linear,
)


__all__ = ["HiggsQuantizationConverter"]


class HiggsQuantizationConverter(Converter):
    """
    Second-pass converter that applies ILP-selected quantization schemes.

    This converter receives an optimized QuantizationConfig (from MSE collector)
    and applies the selected quantization scheme to each layer using the same
    process as model_free_ptq.

    Args:
        optimal_config: QuantizationConfig with config_groups from ILP optimization
        targets: Layer types to quantize (should match MSE collector)
        ignore: Layer patterns to skip (should match MSE collector)
        device: Device for quantization
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

    def validate(self, tensors: Dict[str, torch.Tensor]):
        """Validate we have ILP solution and tensors are quantizable."""
        if not self.optimal_config.config_groups:
            raise ValueError(
                "No config_groups in optimal_config. "
                "Ensure HiggsMSECollectorConverter ran first and solved ILP."
            )

        tensors = split_fused_moe_experts(tensors)
        # Check that we have some quantizable tensors
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

        Uses the same quantization logic as model_free_ptq:
        1. initialize_quantized_linear
        2. calibrate_weight
        3. compress_module
        """
        # Split fused 3D MoE expert tensors into individual 2D expert tensors
        tensors = split_fused_moe_experts(tensors)

        logger.info(
            f"Applying ILP-selected quantization to shard with {len(tensors)} tensors"
        )

        layers_quantized = 0
        lookup_time = 0.0
        quant_time = 0.0

        # Iterate through each config group (scheme)
        for config_group_name, scheme in self.optimal_config.config_groups.items():
            logger.debug(f"Processing config group: {config_group_name}")

            # Find layers matching this scheme's targets
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
                    # SAME quantization logic as model_free_ptq
                    # 1. Initialize module with qparams
                    module = initialize_quantized_linear(weight, scheme, self.device)

                    # 2. Calibrate weight qparams
                    calibrate_weight(module)

                    # 3. Compress module
                    compress_module(module)

                    # 4. Replace in tensor dict
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
                    # Continue with other layers rather than failing completely
                    continue

        logger.info(
            f"Quantized {layers_quantized} layers in this shard "
            f"(lookup: {lookup_time:.2f}s, quant: {quant_time:.2f}s)"
        )

        return tensors

    def create_config(self) -> QuantizationConfig:
        """Return the same optimal config from ILP."""
        return self.optimal_config

    def get_dependencies(self, weight_name: str) -> set[str]:
        """No dependencies for quantization."""
        return set()
