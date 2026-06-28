#!/usr/bin/env python3
"""
Convert safetensors key names to match BF16 model format.

Key transformations:
1. head.weight -> lm_head.weight
2. model.embed_tokens.weight -> embed.weight
3. model.layers.X. -> layers.X.
4. model.layers.X.input_layernorm.weight -> layers.X.attn_norm.weight
5. model.layers.X.mlp.experts.Y.gate_proj.* -> layers.X.ffn.experts.Y.w1.*
6. model.layers.X.mlp.experts.Y.up_proj.* -> layers.X.ffn.experts.Y.w3.*
7. model.layers.X.mlp.experts.Y.down_proj.* -> layers.X.ffn.experts.Y.w2.*
8. model.hc_head.hc_base -> hc_head_base (flatten structure)
9. model.hc_head.hc_fn -> hc_head_fn
10. model.hc_head.hc_scale -> hc_head_scale
11. model.layers.X.attn_hc.base -> layers.X.hc_attn_base (reorder)
12. model.layers.X.attn_hc.fn -> layers.X.hc_attn_fn
13. model.layers.X.attn_hc.scale -> layers.X.hc_attn_scale
14. model.layers.X.ffn_hc.base -> layers.X.hc_ffn_base (reorder)
15. model.layers.X.ffn_hc.fn -> layers.X.hc_ffn_fn
16. model.layers.X.ffn_hc.scale -> layers.X.hc_ffn_scale

This script modifies safetensors files in-place without creating backups.
"""

import argparse
import json
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def convert_key(key: str) -> str | None:
    """Convert a key from quantized format to BF16 format.

    Returns None if the key should be removed.
    """
    # head.weight -> lm_head.weight
    if key == "head.weight":
        return "lm_head.weight"

    # model.embed_tokens.weight -> embed.weight
    if key == "model.embed_tokens.weight":
        return "embed.weight"

    # model.hc_head.hc_base -> hc_head_base (flatten the structure)
    if key.startswith("model.hc_head."):
        # model.hc_head.hc_base -> hc_head_base
        # model.hc_head.hc_fn -> hc_head_fn
        # model.hc_head.hc_scale -> hc_head_scale
        return key.replace("model.hc_head.hc_", "hc_head_")

    # Strip 'model.' prefix from layers
    if key.startswith("model.layers."):
        key = key[6:]  # Remove "model."

        # input_layernorm -> attn_norm
        if ".input_layernorm." in key:
            key = key.replace(".input_layernorm.", ".attn_norm.")

        # attn_hc.base -> hc_attn_base (reorder)
        # layers.0.attn_hc.base -> layers.0.hc_attn_base
        if ".attn_hc." in key:
            key = key.replace(".attn_hc.", ".hc_attn_")

        # ffn_hc.base -> hc_ffn_base (reorder)
        # layers.0.ffn_hc.base -> layers.0.hc_ffn_base
        if ".ffn_hc." in key:
            key = key.replace(".ffn_hc.", ".hc_ffn_")

        # mlp.experts -> ffn.experts
        if ".mlp.experts." in key:
            key = key.replace(".mlp.experts.", ".ffn.experts.")

            # gate_proj -> w1, up_proj -> w3, down_proj -> w2
            key = key.replace(".gate_proj.", ".w1.")
            key = key.replace(".up_proj.", ".w3.")
            key = key.replace(".down_proj.", ".w2.")

    return key


def convert_file(input_path: Path, output_path: Path):
    """Convert a single safetensors file."""
    print(f"Processing {input_path.name}...")

    tensors = {}
    with safe_open(str(input_path), framework="pt") as f:
        for old_key in f.keys():
            new_key = convert_key(old_key)
            if new_key is not None:
                tensors[new_key] = f.get_tensor(old_key)

    save_file(tensors, str(output_path))
    print(f"  Converted {len(tensors)} tensors")


def update_index_file(index_path: Path):
    """Update the model.safetensors.index.json file with new key names."""
    print(f"Processing {index_path.name}...")

    with open(index_path) as f:
        index_data = json.load(f)

    if "weight_map" in index_data:
        new_weight_map = {}
        for old_key, shard_name in index_data["weight_map"].items():
            new_key = convert_key(old_key)
            if new_key is not None:
                new_weight_map[new_key] = shard_name

        index_data["weight_map"] = new_weight_map

    # Write to temp file first, then move to avoid corruption
    temp_path = index_path.with_suffix(".json.tmp")
    with open(temp_path, "w") as f:
        json.dump(index_data, f, indent=2)
    temp_path.replace(index_path)

    print(f"  Updated {len(index_data['weight_map'])} entries")


def main():
    parser = argparse.ArgumentParser(
        description="Convert safetensors key names in-place"
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing safetensors files to convert"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying files"
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir.exists():
        print(f"Error: Directory {model_dir} does not exist")
        return 1

    # Find all safetensors files
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        print(f"No safetensors files found in {model_dir}")
        return 1

    if args.dry_run:
        print("DRY RUN - No files will be modified")
        print(f"\nFound {len(safetensor_files)} files to convert:")
        for f in safetensor_files:
            print(f"  {f.name}")

        # Show sample key conversions
        print("\nSample key conversions from first file:")
        first_file = safetensor_files[0]
        with safe_open(str(first_file), framework="pt") as f:
            for i, old_key in enumerate(f.keys()):
                if i >= 10:
                    break
                new_key = convert_key(old_key)
                if new_key is None:
                    print(f"  {old_key} -> [REMOVED]")
                else:
                    print(f"  {old_key} -> {new_key}")
        return 0

    print(f"Converting {len(safetensor_files)} files in {model_dir}")
    print("WARNING: This will modify files in-place!")
    response = input("Continue? [y/N] ")
    if response.lower() != 'y':
        print("Aborted")
        return 0

    # Process each safetensors file
    for shard_file in safetensor_files:
        # Use a temp file to avoid corruption if the process fails
        temp_path = shard_file.with_suffix(".safetensors.tmp")
        try:
            convert_file(shard_file, temp_path)
            # Replace original with converted file
            temp_path.replace(shard_file)
        except Exception as e:
            print(f"Error processing {shard_file.name}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    # Update index file if it exists
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        update_index_file(index_path)

    print("\nConversion complete!")
    return 0


if __name__ == "__main__":
    exit(main())
