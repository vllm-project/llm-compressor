import os
import torch
import torch.nn as nn
from transformers.models.granitemoehybrid.configuration_granitemoehybrid import GraniteMoeHybridConfig
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridParallelExperts,
    GraniteMoeHybridMoE,
)

from llmcompressor.modeling.moe_context import MoECalibrationModule

from pathlib import Path
from collections import defaultdict
import json
from safetensors.torch import load_file, save_file

class SequentialGraniteMoeExperts(nn.Module):
    """
    Unpacked version of GraniteMoeHybridParallelExperts with individual expert layers.

    This module:
    1. Unpacks the packed expert weights (3D -> individual Linear layers)
    2. Processes experts sequentially
    3. Compatible with FP8 block quantization and vLLM
    """

    def __init__(
        self,
        original: GraniteMoeHybridParallelExperts,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.num_experts = original.num_experts
        self.input_size = original.input_size
        self.output_size = original.output_size
        self.calibrate_all_experts = calibrate_all_experts

        # Create individual linear layers for each expert
        self.experts = nn.ModuleList([
            nn.Linear(self.input_size, self.output_size, bias=False)
            for _ in range(self.num_experts)
        ])

        # Copy weights from the original 3D tensor
        # Original format: [num_experts, output_size, input_size]
        for i in range(self.num_experts):
            self.experts[i].weight.data = original.weight.data[i].clone()

    def forward(self, inputs, expert_size, batch_index=None):
        """
        Forward pass using individual expert layers.
        
        Args:
            inputs: Input tensor to be processed by experts
            expert_size: List containing the size of inputs for each expert
            batch_index: Token indices for routing (needed in calibration mode)
        
        Returns:
            Concatenated output from all experts
        """
        if self.calibrate_all_experts:
            # During calibration, process all inputs through each expert
            # but only keep the outputs corresponding to tokens routed to that expert
            output_list = []
            start_idx = 0
            for i in range(self.num_experts):
                end_idx = start_idx + expert_size[i]
                # Get token indices assigned to this expert
                expert_token_indices = batch_index[start_idx:end_idx]
                # Process ALL tokens through this expert
                expert_out_all = self.experts[i](inputs)
                # Only keep outputs for tokens assigned to this expert
                expert_out = expert_out_all[expert_token_indices]
                output_list.append(expert_out)
                start_idx = end_idx
            results = torch.cat(output_list, dim=0)
        else:
            # Normal routing: only process tokens assigned to this expert
            input_list = inputs.split(expert_size, dim=0)
            output_list = []
            for i in range(self.num_experts):
                output_list.append(self.experts[i](input_list[i]))
            results = torch.cat(output_list, dim=0)
        
        return results


@MoECalibrationModule.register("GraniteMoeHybridMoE")
class CalibrationGraniteMoeHybridMoE(MoECalibrationModule):
    """
    Calibration version of GraniteMoeHybridMoE that unpacks both input_linear and output_linear experts.

    This module:
    1. Replaces both GraniteMoeHybridParallelExperts modules with unpacked versions
    2. Optionally sends all tokens to all experts during calibration
    3. Stays in unpacked form (permanent) for vLLM compatibility and FP8 block quantization
    """

    is_permanent = True

    def __init__(
        self,
        original: GraniteMoeHybridMoE,
        config: GraniteMoeHybridConfig,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.input_size = original.input_size
        self.hidden_size = original.hidden_size
        self.activation = original.activation
        self.calibrate_all_experts = calibrate_all_experts

        # Replace input_linear and output_linear with unpacked versions
        self.input_linear = SequentialGraniteMoeExperts(
            original.input_linear,
            calibrate_all_experts=calibrate_all_experts,
        )
        self.output_linear = SequentialGraniteMoeExperts(
            original.output_linear,
            calibrate_all_experts=calibrate_all_experts,
        )

        # Keep the router unchanged
        self.router = original.router

    def forward(self, layer_input):
        """
        Forward pass of the MoE layer.
        
        Args:
            layer_input: Input tensor of shape [batch_size, seq_len, hidden_size]
        
        Returns:
            Tuple of (output tensor, router_logits) where:
                - output tensor has shape [batch_size, seq_len, hidden_size]
                - router_logits has shape [batch_size * seq_len, num_experts]
        """
        bsz, length, emb_size = layer_input.size()
        layer_input_flat = layer_input.reshape(-1, emb_size)
        
        # Router determines expert assignments
        _, batch_index, batch_gates, expert_size, router_logits = self.router(layer_input_flat)

        if self.calibrate_all_experts:
            # During calibration, send all tokens to all experts
            # Pass batch_index so experts know which outputs to keep
            hidden_states = self.input_linear(layer_input_flat, expert_size, batch_index)
            
            # Apply activation (SwiGLU-style)
            chunked_hidden_states = hidden_states.chunk(2, dim=-1)
            hidden_states = self.activation(chunked_hidden_states[0]) * chunked_hidden_states[1]
            
            # Process through output_linear experts
            expert_outputs = self.output_linear(hidden_states, expert_size, batch_index)
            
            # Apply gating weights
            expert_outputs_gated = expert_outputs * batch_gates[:, None]
        else:
            # Normal routing: only send tokens to assigned experts
            expert_inputs = layer_input_flat[batch_index]
            
            # Process through input_linear experts
            hidden_states = self.input_linear(expert_inputs, expert_size)
            
            # Apply activation (SwiGLU-style)
            chunked_hidden_states = hidden_states.chunk(2, dim=-1)
            hidden_states = self.activation(chunked_hidden_states[0]) * chunked_hidden_states[1]
            
            # Process through output_linear experts
            expert_outputs = self.output_linear(hidden_states, expert_size)
            
            # Apply gating weights
            expert_outputs_gated = expert_outputs * batch_gates[:, None]

        # Aggregate expert outputs
        zeros = torch.zeros(
            (bsz * length, self.input_size),
            dtype=expert_outputs_gated.dtype,
            device=expert_outputs_gated.device
        )
        layer_output = zeros.index_add(0, batch_index, expert_outputs_gated)
        layer_output = layer_output.view(bsz, length, self.input_size)
        
        return layer_output, router_logits


# Legacy function for backward compatibility with prepare.py
def replace(
    config: GraniteMoeHybridConfig,
    module: GraniteMoeHybridMoE,
    calibrate_all_experts: bool,
):
    """
    Legacy replacement function for use with prepare.py.
    
    This function is deprecated. Use moe_calibration_context instead:
    
    Example:
        from llmcompressor.modeling.moe_context import moe_calibration_context
        
        with moe_calibration_context(model, calibrate_all_experts=True):
            # Run calibration
            pass
    
    Args:
        config: The GraniteMoeHybridConfig for the model
        module: The GraniteMoeHybridMoE module to replace
        calibrate_all_experts: Whether to calibrate all experts
    
    Returns:
        CalibrationGraniteMoeHybridMoE calibration module
    """
    return CalibrationGraniteMoeHybridMoE(
        module,
        config,
        calibrate_all_experts=calibrate_all_experts,
    )


def replace_granite_moe_with_linear_experts(model):
    """
    Legacy replacement function that recursively replaces all GraniteMoeHybridMoE modules.
    
    This function is deprecated. Use moe_calibration_context instead:
    
    Example:
        from llmcompressor.modeling.moe_context import moe_calibration_context
        
        with moe_calibration_context(model, calibrate_all_experts=True):
            # Run calibration
            pass
    
    Args:
        model: The model containing GraniteMoeHybridMoE modules
    
    Returns:
        The modified model with replaced expert modules
    """
    def replace_moe_modules(module, name=''):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if child.__class__.__name__ == 'GraniteMoeHybridMoE':
                # Create replacement module with unpacked experts
                calibrated = CalibrationGraniteMoeHybridMoE(
                    original=child,
                    config=model.config,
                    calibrate_all_experts=True,
                )
                # Replace the module
                setattr(module, child_name, calibrated)
                print(f"Replaced {full_name}: GraniteMoeHybridMoE with unpacked experts")
            else:
                # Recursively process children
                replace_moe_modules(child, full_name)
     
    replace_moe_modules(model)
    return model


def pack_3d_experts(source_dir):
    """
    Transform Granite MoE model from per-expert storage to stacked 3D tensor storage
    
    From: model.layers.{L}.block_sparse_moe.{linear_type}.experts.{E}.{param}
    To:   model.layers.{L}.block_sparse_moe.{linear_type}.{param}
    
    """
    source_dir = Path(source_dir)
    
    # Load the index file
    index_file = source_dir / "model.safetensors.index.json"
    with open(index_file, "r") as f:
        index_data = json.load(f)
    
    weight_map = index_data["weight_map"]
    
    # Group tensors by layer, linear type, and parameter
    # Structure: {(layer_num, linear_type, param): {expert_num: (tensor_name, file_name)}}
    grouped_tensors = defaultdict(dict)
    other_tensors = {}  # Non-expert tensors (router, embeddings, etc.)
    
    for tensor_name, file_name in weight_map.items():
        # Check if this is an expert tensor
        # Pattern: model.layers.{L}.block_sparse_moe.{linear_type}.experts.{E}.{param}
        if ".block_sparse_moe." in tensor_name and ".experts." in tensor_name:
            parts = tensor_name.split(".")
            
            try:
                # Find the indices of key parts
                layers_idx = parts.index("layers")
                layer_num = int(parts[layers_idx + 1])
                
                experts_idx = parts.index("experts")
                expert_num = int(parts[experts_idx + 1])
                
                # The linear type is right before "experts"
                # e.g., "input_linear" or "output_linear"
                linear_type = parts[experts_idx - 1]
                
                # The parameter is after expert number
                # e.g., "weight" or "weight_scale"
                param = ".".join(parts[experts_idx + 2:])
                
                # Create grouping key
                group_key = (layer_num, linear_type, param)
                grouped_tensors[group_key][expert_num] = (tensor_name, file_name)
                
            except (ValueError, IndexError) as e:
                # If parsing fails, treat as other tensor
                print(f"  Warning: Could not parse expert tensor: {tensor_name}")
                other_tensors[tensor_name] = file_name
        else:
            other_tensors[tensor_name] = file_name
    
    # Load all safetensors files
    print("Loading source safetensors files...")
    loaded_tensors = {}
    unique_files = set(weight_map.values())
    old_files = list(unique_files)  # Store list of old files to delete later
    
    for file_name in unique_files:
        file_path = source_dir / file_name
        print(f"  Loading {file_name}...")
        loaded_tensors[file_name] = load_file(str(file_path))
    
    # Create new tensors by stacking experts
    print("\nStacking expert tensors...")
    new_tensors = {}
    
    # Process each grouped tensor
    for (layer_num, linear_type, param), experts_dict in sorted(grouped_tensors.items()):
        print(f"  Processing layer {layer_num}, {linear_type}.{param}...")
        
        # Get all expert tensors for this group
        expert_nums = sorted(experts_dict.keys())
        expert_tensors = []
        
        for expert_num in expert_nums:
            tensor_name, file_name = experts_dict[expert_num]
            tensor = loaded_tensors[file_name][tensor_name]
            expert_tensors.append(tensor)
        
        # Stack along first dimension to create 3D tensor
        stacked_tensor = torch.stack(expert_tensors, dim=0)
        
        # Create new tensor name (remove .experts.{E} part)
        new_tensor_name = f"model.layers.{layer_num}.block_sparse_moe.{linear_type}.{param}"
        new_tensors[new_tensor_name] = stacked_tensor
        
        print(f"    {new_tensor_name}: {list(stacked_tensor.shape)} (stacked {len(expert_tensors)} experts)")
    
    # Copy non-expert tensors (router, embeddings, etc.)
    print("\nCopying non-expert tensors...")
    for tensor_name, file_name in other_tensors.items():
        tensor = loaded_tensors[file_name][tensor_name]
        new_tensors[tensor_name] = tensor
        print(f"  Copied: {tensor_name}")
    
    # Determine file distribution for new tensors
    # Simple strategy: distribute roughly equally across same number of files
    num_output_files = len(unique_files)
    tensors_list = list(new_tensors.items())
    
    # Calculate approximate size per file
    total_numel = sum(t.numel() * t.element_size() for _, t in tensors_list)
    target_size_per_file = total_numel / num_output_files
    
    # Distribute tensors across files
    print(f"\nDistributing tensors across {num_output_files} files...")
    file_tensors = [{}  for _ in range(num_output_files)]
    file_sizes = [0] * num_output_files
    new_weight_map = {}
    
    for tensor_name, tensor in tensors_list:
        # Find file with smallest current size
        min_idx = file_sizes.index(min(file_sizes))
        file_tensors[min_idx][tensor_name] = tensor
        file_sizes[min_idx] += tensor.numel() * tensor.element_size()
        
        # Update weight map
        file_name = f"model-{min_idx+1:05d}-of-{num_output_files:05d}.safetensors"
        new_weight_map[tensor_name] = file_name
    
    # Save new safetensors files with temporary names
    print("\nSaving new safetensors files (temporary)...")
    temp_files = []
    for i, tensors_dict in enumerate(file_tensors):
        if tensors_dict:  # Only save if not empty
            file_name = f"model-{i+1:05d}-of-{num_output_files:05d}.safetensors"
            temp_file_name = f"model-{i+1:05d}-of-{num_output_files:05d}.safetensors.tmp"
            output_path = source_dir / temp_file_name
            print(f"  Saving {temp_file_name} ({len(tensors_dict)} tensors)...")
            save_file(tensors_dict, str(output_path))
            temp_files.append((temp_file_name, file_name))
    
    # Save updated index file with temporary name
    print("\nSaving updated index file (temporary)...")
    new_index_data = {
        "metadata": index_data.get("metadata", {}),
        "weight_map": new_weight_map
    }
    
    temp_index_file = source_dir / "model.safetensors.index.json.tmp"
    with open(temp_index_file, "w") as f:
        json.dump(new_index_data, f, indent=2)
    
    # Now delete old files
    print("\nDeleting old safetensors files...")
    for old_file in old_files:
        old_file_path = source_dir / old_file
        if old_file_path.exists():
            old_file_path.unlink()
            print(f"  Deleted {old_file}")
    
    # Delete old index file
    if index_file.exists():
        index_file.unlink()
        print(f"  Deleted model.safetensors.index.json")
    
    # Rename temporary files to final names
    print("\nRenaming temporary files to final names...")
    for temp_name, final_name in temp_files:
        temp_path = source_dir / temp_name
        final_path = source_dir / final_name
        temp_path.rename(final_path)
        print(f"  Renamed {temp_name} -> {final_name}")
    
    # Rename temporary index file
    temp_index_file.rename(index_file)
    print(f"  Renamed model.safetensors.index.json.tmp -> model.safetensors.index.json")
    
    # Update config.json to rename mamba layers to mixer
    print("\nUpdating config.json to rename mamba layers to mixer...")
    config_file = source_dir / "config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            config_data = json.load(f)
        
        # Check if quantization_config exists and has ignore list
        if "quantization_config" in config_data and "ignore" in config_data["quantization_config"]:
            ignore_list = config_data["quantization_config"]["ignore"]
            updated_count = 0
            
            # Replace mamba.in_proj with mixer.in_proj and mamba.out_proj with mixer.out_proj
            for i, entry in enumerate(ignore_list):
                if "mamba.in_proj" in entry or "mamba.out_proj" in entry:
                    new_entry = entry.replace("mamba.in_proj", "mixer.in_proj").replace("mamba.out_proj", "mixer.out_proj")
                    ignore_list[i] = new_entry
                    updated_count += 1
                    print(f"  Updated: {entry} -> {new_entry}")
            
            # Save updated config
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)
            
            print(f"  Updated {updated_count} entries in config.json")
        else:
            print("  No quantization_config.ignore found in config.json")
    else:
        print("  config.json not found")
    
    # Print summary
    num_stacked = len(grouped_tensors)
    num_other = len(other_tensors)
    print(f"\n📊 Summary:")
    print(f"   Stacked expert groups: {num_stacked}")
    print(f"   Non-expert tensors: {num_other}")
    print(f"\nCheckpoint Updated for vLLM Compatibility")

            
class GraniteMoeHybridParallelExpertsLinear(torch.nn.Linear):
    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None:
        """Use a real Linear so that llmcompressor and vllm can handle it easier.
        1. Change .weight from 3D [num_experts, output_size, input_size] to 2D
            [num_experts * output_size, input_size] before calling llm-compressor
        2. Change it back to 3D before saving ckpt
        """
        super().__init__(
            input_size, output_size * num_experts, bias=False, device="meta"
        )
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.is_2d: bool = True

    @classmethod
    def from_3d_expert(cls, original: GraniteMoeHybridParallelExperts):
        """Reshape weights of GraniteMoeHybridParallelExperts module into 2D and store
        them as weights of this "Linear" module.
        """
        newMoeLin = cls(original.num_experts, original.input_size, original.output_size)
        newMoeLin.weight = torch.nn.Parameter(
            original.weight.view(-1, original.input_size).clone(),
            requires_grad=False,
        )
        original.to("cpu")
        newMoeLin.is_2d = True
        return newMoeLin

    def to_3d_expert(self) -> None:
        """Convert weights and quantization parameters from 2D to 3D shape."""
        dim0_mul = self.num_experts * self.output_size
        assert (
            self.weight.shape == torch.Size((dim0_mul, self.input_size))
            and hasattr(self, "weight_scale")
            and self.weight_scale.shape == torch.Size((dim0_mul, 1))
        ), "Shape mismatch, please check."

        self.weight = torch.nn.Parameter(
            self.weight.view(
                self.num_experts, self.output_size, self.input_size
            ).clone(),
            requires_grad=False,
        )
        self.weight_scale = torch.nn.Parameter(
            self.weight_scale.view(self.num_experts, self.output_size, 1).clone(),
            requires_grad=False,
        )
        if hasattr(self, "weight_zero_point"):
            assert self.weight_zero_point.shape == torch.Size((dim0_mul, 1))
            self.weight_zero_point = torch.nn.Parameter(
                self.weight_zero_point.view(
                    self.num_experts, self.output_size, 1
                ).clone(),
                requires_grad=False,
            )
        self.is_2d = False

    def forward(self, inputs, expert_size):
        """Modified from original forward()"""

        input_list = inputs.split(expert_size, dim=0)

        weight_3d = self.weight.view(
            self.num_experts, self.output_size, self.input_size
        )
        output_list = []
        for i in range(self.num_experts):
            output_list.append(torch.nn.functional.linear(input_list[i], weight_3d[i]))

        results = torch.cat(output_list, dim=0)
        return results

    def __repr__(self):
        if self.is_2d:
            sizes_str = f"(out={self.weight.shape[0]},in={self.weight.shape[1]})"
        else:
            sizes_str = (
                f"(exp={self.weight.shape[0]},out={self.weight.shape[1]},"
                f"in={self.weight.shape[2]})"
            )
        return f"{self.__class__.__name__}{sizes_str}"
