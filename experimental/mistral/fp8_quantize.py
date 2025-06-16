import argparse
import os
import json
import torch
import safetensors.torch

def per_tensor_quantize(tensor):
    """Quantize a tensor to FP8 using per-tensor static scaling factor."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    if tensor.numel() == 0:
        min_val, max_val = torch.tensor(-16.0, dtype=tensor.dtype), torch.tensor(16.0, dtype=tensor.dtype)
    else:
        min_val, max_val = tensor.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs())
    scale = finfo.max / amax.clamp(min=1e-12)
    qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max).to(torch.float8_e4m3fn)
    scale = scale.float().reciprocal()
    return qweight, scale

def is_quantizable(name):
    """Check if the tensor name indicates it can be quantized."""
    return name.startswith('layers.') and name.endswith(('.wk.weight', '.wo.weight', '.wq.weight', '.wv.weight', '.w1.weight', '.w2.weight', '.w3.weight'))

def process_safetensors_file(file_path):
    """Process a single safetensors file in-place, quantizing weights to FP8."""
    print(f"Processing {file_path}")
    tensors = safetensors.torch.load_file(file_path)
    
    modified_tensors = {}
    for name, tensor in tensors.items():
        if is_quantizable(name):
            print("Quantizing", name)
            qweight, scale = per_tensor_quantize(tensor)
            modified_tensors[name] = qweight
            modified_tensors[f"{name[:-len("weight")]}qscale_weight"] = scale
        else:
            modified_tensors[name] = tensor

    safetensors.torch.save_file(modified_tensors, file_path)
    print(f"Updated {file_path} with quantized tensors")

def update_index_file(index_file_path):
    """Update the index file for the quantized model."""
    print(f"Updating index file: {index_file_path}")
    with open(index_file_path, 'r') as f:
        index = json.load(f)
    
    new_weight_map = {}
    for tensor_name, file_name in index['weight_map'].items():
        new_weight_map[tensor_name] = file_name
        if is_quantizable(tensor_name):
            new_weight_map[f"{tensor_name[:-len("weight")]}qscale_weight"] = file_name
    
    index['weight_map'] = new_weight_map
    
    # Recalculate total_size
    total_size = sum(os.path.getsize(os.path.join(os.path.dirname(index_file_path), file)) 
                     for file in set(index['weight_map'].values()))
    index['metadata']['total_size'] = total_size
    
    with open(index_file_path, 'w') as f:
        json.dump(index, f, indent=2)
    print(f"Updated index file {index_file_path}")

def update_config(config_file_path):
    """Update the params.json file for the quantized model."""
    print(f"Updating config file: {config_file_path}")
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    config["quantization"] = {
        "config_groups": {
        "group_0": {
            "input_activations": {
                "dynamic": True,
                "num_bits": 8,
                "observer": None,
                "strategy": "token",
                "symmetric": True,
                "type": "float"
            },
            "targets": ["Linear"],
            "weights": {
                "dynamic": False,
                "num_bits": 8,
                "observer": "minmax",
                "strategy": "tensor",
                "symmetric": True,
                "type": "float"
            }
        }},
        "format": "float-quantized",
        "ignore": ["lm_head", "output"],
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed"
    }
    
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Updated config file {config_file_path}")

def process_directory(directory):
    """Process all safetensors files in the given directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.safetensors'):
            process_safetensors_file(file_path)
        elif filename == 'consolidated.safetensors.index.json':
            update_index_file(file_path)
        elif filename == 'params.json':
            update_config(file_path)
        else:
            print(f"Skipping unrecognized file: {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert mistral safetensors model to FP8 in-place.')
    parser.add_argument('directory', type=str, help='The directory containing the safetensors files and index file.')
    
    args = parser.parse_args()
    process_directory(args.directory)