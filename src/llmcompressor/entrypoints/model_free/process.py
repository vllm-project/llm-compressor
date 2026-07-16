import os
import re
from collections import defaultdict
from typing import Iterable

import torch
from compressed_tensors.compressors import compress_module
from compressed_tensors.entrypoints.convert import Converter
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.utils import match_quantizable_tensors
from compressed_tensors.utils.safetensors_load import (
    InverseWeightMap,
    load_tensors_from_inverse_weight_map,
)
from loguru import logger
from safetensors.torch import save_file
from torch.nn import Module

from llmcompressor.entrypoints.model_free.lifecycle import (
    calibrate_weight,
    initialize_quantized_linear,
    validate_weight_for_quantization,
)
from llmcompressor.entrypoints.model_free.microscale import (
    get_fused_names,
    is_microscale_scheme,
    DEFAULT_FUSED_MAPPINGS,
)
from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    freeze_module_quantization,
    initialize_observer,
    observe,
    update_qparams,
)
from llmcompressor.observers import FusionHandler

__all__ = [
    "ModelFreePtqConverter",
    "split_fused_moe_experts",
]

class ModelFreePtqConverter():
    def __init__(self, scheme, ignore, converter=None):
        self.scheme = scheme
        self.ignore = ignore
        self.converter = converter
        self.is_microscale = is_microscale_scheme(scheme)

    def process(self, tensors):
        return tensors

    def validate(self, tensors):
        pass

    def create_config(self):
        return None
    
    def validate_file(self, inverse_weight_map, save_path, device):
        device = torch.device("meta")
        tensors = load_tensors_from_inverse_weight_map(inverse_weight_map, device)

        if self.converter is not None:
            self.converter.validate(tensors)

        for _, name in match_quantizable_tensors(tensors, self.ignore, self.scheme.targets):
            validate_weight_for_quantization(tensors[name], self.scheme, name)

    def process_file(self, inverse_weight_map, save_path, device) -> tuple[int, dict[str, str]]:
        tensors = load_tensors_from_inverse_weight_map(inverse_weight_map, device)
        tensors = split_fused_moe_experts(tensors)

        if self.converter is not None:
            tensors = self.converter.process(tensors)

        # Microscale only... fused lookup
        if self.is_microscale:
            fused_sets, _ = get_fused_names(list(tensors.keys()))
            fused_name_to_fused_index: dict[str, int] = {
                name: index
                for index, matched_set in enumerate(fused_sets)
                for name in matched_set.values()
                if name is not None
            }
            fused_modules: dict[int, dict[str, Module]] = defaultdict(dict)

        # Per-tensor loop                
        for module_name, name in match_quantizable_tensors(tensors, self.ignore, self.scheme.targets):
            validate_weight_for_quantization(tensors[name], self.scheme, name)
            module = initialize_quantized_linear(tensors[name], self.scheme, device)
    
            # Defer fused tensors for processing down the line
            if self.is_microscale and name in fused_name_to_fused_index:
                fused_index = fused_name_to_fused_index[name]
                fused_modules[fused_index][name] = module
                initialize_observer(module, "weight")
                apply_calibration_status(module)
                
                continue

            # Standard path
            calibrate_weight(module)
            compress_module(module)

            del tensors[name]
            prefix = module_name + '.'
            for key, value in module.state_dict(prefix=prefix).items():
                tensors[key] = value.to("cpu")
        
        # Only for microscale compressed fused models with a shared global state
        if self.is_microscale:
            for named_modules in fused_modules.values():
                FusionHandler.fuse(
                    [(mod.weight_observer, mod) for mod in named_modules.values()]
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

        # Save ALL tensors to this shard's output — including partner tensors fetched
        # from other shards. Partners are re-saved here so future runs don't need to
        # re-fetch them. The caller updates the safetensors index to reflect new locations.
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        save_file(tensors, save_path)
        total_size = sum(t.nbytes for t in tensors.values())
        weight_map = {key: os.path.basename(save_path) for key in tensors.keys()}
        return total_size, weight_map
        

    def get_dependencies(self, weight_name: str) -> set[str]:
        deps = set()
        if self.is_microscale:
            for primary_pattern, partner_templates in DEFAULT_FUSED_MAPPINGS.items():
                match = re.match(primary_pattern, weight_name)
                if match is None:
                    continue

                # Build partner names using named groups from the match
                for partner_template in partner_templates:
                    partner_name = partner_template.format(**match.groupdict())

                    deps.add(partner_name)
        return deps


def split_fused_moe_experts(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Find fused MoE experts (with gate_up_proj/down_proj).
    Split them from 3D tensors into individual 2D expert tensors.

    Args:
        tensors: Dictionary of loaded tensors from safetensors file

    Returns:
        split_tensors: New dictionary with split expert weights
    """
    split_tensors = {}

    params_to_split = {
        # If a 3D gate_up_proj layer is found, split it into a
        # 2D gate_proj and up_proj layer for each expert
        "gate_up_proj": ["gate_proj", "up_proj"],
        # If a 3D down_proj layer is found, split it into a
        # 2D down_proj layer for each expert
        "down_proj": ["down_proj"],
    }

    for name, tensor in tensors.items():
        keys_to_split = [key for key in params_to_split if key in name]
        if len(keys_to_split) >= 2:
            raise ValueError(f"Found multiple keys matching {name}: {keys_to_split}")

        elif len(keys_to_split) == 1 and tensor.ndim == 3:
            unsplit_name = keys_to_split[0]
            split_names = params_to_split[unsplit_name]

            # Get number of experts
            num_experts = tensor.shape[0]

            if tensor.shape[1] % len(split_names) != 0:
                raise ValueError(
                    f"{unsplit_name} expects a second dimension divisible by "
                    f"{len(split_names)} but got shape: {tensor.shape}"
                )

            # Split into experts
            intermediate_size = tensor.shape[1] // len(split_names)
            for expert_idx in range(num_experts):
                expert_tensor = tensor[expert_idx]
                # Split into layers
                split_layers = expert_tensor.split(intermediate_size, dim=0)
                for split_name, split_layer in zip(split_names, split_layers):
                    key = name.replace(unsplit_name, f"{expert_idx}.{split_name}")
                    if not key.endswith(".weight"):
                        key = f"{key}.weight"
                    split_tensors[key] = split_layer

            logger.info(f"Split {name} into {num_experts} experts")

        else:
            # Non-MoE or non-3D tensors, keep as is
            split_tensors[name] = tensor

    return split_tensors
