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
from datetime import datetime

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


def pack_3d_experts(
    source_dir,
    validate=True,
    backup=True,
    allow_missing_experts=False,
    verbose=True
):
    """
    Transform MoE model from per-expert storage to 3D stacked tensors.
    
    From: model.layers.{L}.block_sparse_moe.{linear_type}.experts.{E}.{param}
    To:   model.layers.{L}.block_sparse_moe.{linear_type}.{param}
    
    Args:
        source_dir: Model directory path
        validate: Validate shapes and expert continuity
        backup: Create backup before modification (RECOMMENDED)
        allow_missing_experts: Don't fail if some experts are missing
        verbose: Print progress messages
    """
    source_dir = Path(source_dir)
    index_file = source_dir / "model.safetensors.index.json"
    backup_dir = None
    temp_files = []
    
    def log(msg):
        if verbose: print(msg)
    
    try:
        # === BACKUP ===
        if backup:
            backup_dir = source_dir.parent / f"{source_dir.name}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True)
            for f in source_dir.glob("*.safetensors*"):
                shutil.copy2(f, backup_dir / f.name)
            log(f"✓ Backup created at {backup_dir}")
        
        # === LOAD INDEX ===
        with open(index_file) as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
        
        # === GROUP TENSORS ===
        grouped = defaultdict(dict)  # {(layer, linear_type, param): {expert_num: (name, file)}}
        other = {}
        
        for name, file in weight_map.items():
            if ".block_sparse_moe." in name and ".experts." in name:
                parts = name.split(".")
                try:
                    layer = int(parts[parts.index("layers") + 1])
                    expert = int(parts[parts.index("experts") + 1])
                    linear_type = parts[parts.index("experts") - 1]
                    param = ".".join(parts[parts.index("experts") + 2:])
                    grouped[(layer, linear_type, param)][expert] = (name, file)
                except (ValueError, IndexError):
                    other[name] = file
            else:
                other[name] = file
        
        log(f"✓ Found {len(grouped)} expert groups, {len(other)} other tensors")
        
        # === LOAD FILES ===
        log("Loading files...")
        loaded = {}
        old_files = set(weight_map.values())
        for file in old_files:
            loaded[file] = load_file(str(source_dir / file))
        
        # === STACK EXPERTS ===
        log("Stacking experts...")
        new_tensors = {}
        
        for (layer, linear_type, param), experts in sorted(grouped.items()):
            expert_nums = sorted(experts.keys())
            
            # Validate
            if validate:
                # Check continuity
                expected = list(range(len(expert_nums)))
                if expert_nums != expected:
                    missing = set(expected) - set(expert_nums)
                    if missing and not allow_missing_experts:
                        raise ValueError(f"Missing experts {missing} in layer {layer}, {linear_type}.{param}")
                
                # Check shapes and dtypes
                shapes = [loaded[experts[e][1]][experts[e][0]].shape for e in expert_nums]
                dtypes = [loaded[experts[e][1]][experts[e][0]].dtype for e in expert_nums]
                if len(set(shapes)) > 1:
                    raise ValueError(f"Shape mismatch in layer {layer}, {linear_type}.{param}: {set(shapes)}")
                if len(set(dtypes)) > 1:
                    raise ValueError(f"Dtype mismatch in layer {layer}, {linear_type}.{param}: {set(dtypes)}")
            
            # Stack
            tensors = [loaded[experts[e][1]][experts[e][0]] for e in expert_nums]
            stacked = torch.stack(tensors, dim=0)
            new_name = f"model.layers.{layer}.block_sparse_moe.{linear_type}.{param}"
            new_tensors[new_name] = stacked
            log(f"  Layer {layer} {linear_type}.{param}: {list(stacked.shape)}")
        
        # Copy other tensors
        for name, file in other.items():
            new_tensors[name] = loaded[file][name]
        
        # === DISTRIBUTE ACROSS FILES ===
        log("Distributing tensors...")
        num_files = len(old_files)
        tensor_sizes = [(n, t.numel() * t.element_size()) for n, t in new_tensors.items()]
        tensor_sizes.sort(key=lambda x: x[1], reverse=True)
        
        file_tensors = [{} for _ in range(num_files)]
        file_sizes = [0] * num_files
        new_weight_map = {}
        
        for name, size in tensor_sizes:
            min_idx = file_sizes.index(min(file_sizes))
            file_tensors[min_idx][name] = new_tensors[name]
            file_sizes[min_idx] += size
            new_weight_map[name] = f"model-{min_idx+1:05d}-of-{num_files:05d}.safetensors"
        
        # === SAVE FILES (TEMP) ===
        log("Saving files...")
        saved_files = []
        for i, tensors in enumerate(file_tensors):
            if tensors:
                file_name = f"model-{i+1:05d}-of-{num_files:05d}.safetensors"
                temp_name = f"{file_name}.tmp"
                temp_path = source_dir / temp_name
                save_file(tensors, str(temp_path))
                temp_files.append(temp_path)
                saved_files.append((temp_name, file_name))
        
        # Save index (temp)
        temp_index = source_dir / "model.safetensors.index.json.tmp"
        with open(temp_index, "w") as f:
            json.dump({"metadata": index_data.get("metadata", {}), "weight_map": new_weight_map}, f, indent=2)
        temp_files.append(temp_index)
        
        # === FINALIZE (DELETE OLD, RENAME TEMP) ===
        log("Finalizing...")
        # Delete old
        for old in old_files:
            (source_dir / old).unlink()
        index_file.unlink()
        
        # Rename temp
        for temp, final in saved_files:
            (source_dir / temp).rename(source_dir / final)
        temp_index.rename(index_file)
        temp_files.clear()
        
        # === VERIFY ===
        if validate:
            with open(index_file) as f:
                check = json.load(f)
            remaining_experts = [n for n in check["weight_map"] if ".experts." in n]
            if remaining_experts:
                raise ValueError(f"Verification failed: {len(remaining_experts)} unpacked experts remain")
        
        log(f"✓ Success! Transformed {len(grouped)} expert groups")
        
    except Exception as e:
        log(f"✗ Error: {e}")
        
        # === ROLLBACK ===
        if backup and backup_dir and backup_dir.exists():
            log("Rolling back...")
            for temp in temp_files:
                if temp.exists(): temp.unlink()
            for f in source_dir.glob("*.safetensors*"):
                f.unlink()
            for f in backup_dir.glob("*"):
                shutil.copy2(f, source_dir / f.name)
            log("✓ Rolled back to backup")
        
        raise
    
    finally:
        # Cleanup temp files
        for temp in temp_files:
            if temp.exists(): temp.unlink()

            
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
