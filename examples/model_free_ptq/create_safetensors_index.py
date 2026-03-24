"""
Script to create a model.safetensors.index.json for a single model.safetensors file.

Usage:
    python create_safetensors_index.py <model_dir>
"""

import argparse
import json
import os
import struct


def get_safetensors_metadata(path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
    header = json.loads(header_bytes)
    return header


def create_index(model_dir):
    safetensors_file = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(safetensors_file):
        raise FileNotFoundError(f"No model.safetensors found in {model_dir}")

    header = get_safetensors_metadata(safetensors_file)

    total_size = 0
    weight_map = {}

    for key, info in header.items():
        if key == "__metadata__":
            continue
        dtype_sizes = {
            "F64": 8,
            "F32": 4,
            "F16": 2,
            "BF16": 2,
            "I64": 8,
            "I32": 4,
            "I16": 2,
            "I8": 1,
            "U8": 1,
            "BOOL": 1,
            "F8_E4M3": 1,
            "F8_E5M2": 1,
        }
        dtype = info["dtype"]
        shape = info["shape"]
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        element_size = dtype_sizes.get(dtype, 1)
        total_size += num_elements * element_size
        weight_map[key] = "model.safetensors"

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }

    out_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Written {out_path} with {len(weight_map)} tensors, total_size={total_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Directory containing model.safetensors")
    args = parser.parse_args()
    create_index(args.model_dir)
