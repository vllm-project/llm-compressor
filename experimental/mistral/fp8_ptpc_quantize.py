import argparse
import os
import json
import torch
import safetensors.torch


def is_quantizable(name):
    """Check if the tensor name indicates it can be quantized."""
    return name.startswith('layers.') and name.endswith(('.wk.weight', '.wo.weight', '.wq.weight', '.wv.weight', '.w1.weight', '.w2.weight', '.w3.weight'))

def channelwise_quantize(tensor: torch.Tensor):
    """Quantize a tensor to FP8 (E4M3FN) using per-channel static scaling.

    The scaling factor is computed independently for each slice along ``dim``
    (for transformer linear weights this is the *output* channel / first
    dimension).  The function returns the quantized tensor and the *inverse*
    of the scale (i.e. the value that should be multiplied with the de-
    quantized FP8 numbers to recover FP32).
    """
    assert tensor.numel() != 0
    finfo = torch.finfo(torch.float8_e4m3fn)
    min_val, max_val = tensor.aminmax(dim=1, keepdims=True)
    amax = torch.maximum(min_val.abs(), max_val.abs())
    # scale = amax / finfo.max
    scale = amax / (float(finfo.max - finfo.min) / 2)
    scale = torch.clamp(scale, min=torch.finfo(torch.float32).eps)
    qweight = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(torch.float8_e4m3fn)
    scale = scale
    return qweight, scale


def process_safetensors_file(file_path: str):
    """Quantize eligible weights in a safetensors file in-place (channelwise)."""
    print(f"Processing {file_path}")
    tensors = safetensors.torch.load_file(file_path)

    modified_tensors = {}
    for name, tensor in tensors.items():
        modified_tensors[name] = tensor
        if is_quantizable(name):
            print("Quantizing", name)
            qweight, scale = channelwise_quantize(tensor)
            modified_tensors[name] = qweight
            scale_name = f"{name[:-len('weight')]}qscale_weight"
            modified_tensors[scale_name] = scale

    safetensors.torch.save_file(modified_tensors, file_path)
    print(f"Updated {file_path} with channelwise-quantized tensors")


def update_index_file(index_file_path: str):
    """Ensure the weight map reflects additional qscale tensors."""
    print(f"Updating index file: {index_file_path}")
    with open(index_file_path, "r") as f:
        index = json.load(f)

    new_weight_map = {}
    for tensor_name, file_name in index["weight_map"].items():
        new_weight_map[tensor_name] = file_name
        if is_quantizable(tensor_name):
            scale_name = f"{tensor_name[:-len('weight')]}qscale_weight"
            new_weight_map[scale_name] = file_name

    index["weight_map"] = new_weight_map

    # Re‑compute total size
    total_size = sum(
        os.path.getsize(os.path.join(os.path.dirname(index_file_path), f))
        for f in set(new_weight_map.values())
    )
    index.setdefault("metadata", {})["total_size"] = total_size

    with open(index_file_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Index file {index_file_path} updated")


def update_config(config_file_path: str):
    """Annotate params.json with quantization metadata."""
    print(f"Updating config file: {config_file_path}")
    with open(config_file_path, "r") as f:
        config = json.load(f)

    config["quantization"] = {
        "config_groups": {
        "group_0": {
            "input_activations": {
                "actorder": None,
                "block_structure": None,
                "dynamic": True,
                "group_size": None,
                "num_bits": 8,
                "observer": None,
                "observer_kwargs": {},
                "strategy": "token",
                "symmetric": True,
                "type": "float"
            },
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": None,
                "num_bits": 8,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": "channel",
                "symmetric": True,
                "type": "float"
            }
        }},
        "format": "float-quantized",
        "ignore": ["lm_head", "output"],
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed"
    }

    with open(config_file_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config file {config_file_path} updated")


def process_directory(directory: str):
    """Walk the directory and process each relevant file."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".safetensors"):
            process_safetensors_file(file_path)
        elif filename == "consolidated.safetensors.index.json":
            update_index_file(file_path)
        elif filename == "params.json":
            update_config(file_path)
        else:
            print(f"Skipping unrecognised file: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Mistral safetensors model weights to FP8 (channelwise) in-place."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing the safetensors shards and index file.",
    )
    args = parser.parse_args()
    process_directory(args.directory)
