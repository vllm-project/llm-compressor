import torch
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
from safetensors.torch import load_file, save_file

from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridParallelExperts,
)


#for fp8 block quantization
def replace_granite_moe_with_linear_experts(model):
    """
    Convert GraniteMoeHybridParallelExperts modules into individual expert layers.
    Each expert will be stored as a separate nn.Linear module.
    """
    
    class SeparatedExperts(nn.Module):
        """Replacement module with individual expert linear layers"""
        def __init__(self, num_experts, input_size, output_size, original_weight):
            super().__init__()
            self.num_experts = num_experts
            self.input_size = input_size
            self.output_size = output_size
            
            # Create individual linear layers for each expert
            self.experts = nn.ModuleList([
                nn.Linear(input_size, output_size, bias=False)
                for _ in range(num_experts)
            ])
            
            # Copy weights from the original 3D tensor
            # Original format: [num_experts, output_size, input_size]
            for i in range(num_experts):
                self.experts[i].weight.data = original_weight[i].clone()
        
        def forward(self, inputs, expert_size):
            """Forward pass using individual expert layers"""
            input_list = inputs.split(expert_size, dim=0)
            output_list = []
            for i in range(self.num_experts):
                output_list.append(self.experts[i](input_list[i]))
            results = torch.cat(output_list, dim=0)
            return results
    
    # Find and replace all GraniteMoeHybridParallelExperts modules
    def replace_parallel_experts(module, name=''):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if child.__class__.__name__ == 'GraniteMoeHybridParallelExperts':
                # Create replacement module with separated experts
                separated = SeparatedExperts(
                    num_experts=child.num_experts,
                    input_size=child.input_size,
                    output_size=child.output_size,
                    original_weight=child.weight.data
                )
                # Replace the module
                setattr(module, child_name, separated)
                print(f"Replaced {full_name}: {child.num_experts} experts, "
                      f"input_size={child.input_size}, output_size={child.output_size}")
            else:
                # Recursively process children
                replace_parallel_experts(child, full_name)
    
    replace_parallel_experts(model)
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
