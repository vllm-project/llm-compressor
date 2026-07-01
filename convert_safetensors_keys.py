#!/usr/bin/env python3
"""
Convert safetensors key names to match BF16 model format.

Key transformations:
1. head.weight -> lm_head.weight
2. model.embed_tokens.weight -> embed.weight
3. model.norm.* -> norm.*
4. model.layers.X. -> layers.X.
5. layers.X.input_layernorm.weight -> layers.X.attn_norm.weight
6. layers.X.post_attention_layernorm.weight -> layers.X.ffn_norm.weight
7. layers.X.self_attn.* -> layers.X.attn.*
8. layers.X.attn.q_a_norm.weight -> layers.X.attn.q_norm.weight
9. layers.X.attn.compressor.kv_norm.weight -> layers.X.attn.compressor.norm.weight
10. layers.X.attn.compressor.indexer.kv_norm.weight -> layers.X.attn.indexer.compressor.norm.weight
11. layers.X.attn.sinks -> layers.X.attn.attn_sink
12. layers.X.attn.kv_proj.* -> layers.X.attn.wkv.*
13. layers.X.attn.o_a_proj.* -> layers.X.attn.wo_a.*
14. layers.X.attn.o_b_proj.* -> layers.X.attn.wo_b.*
15. layers.X.attn.q_a_proj.* -> layers.X.attn.wq_a.*
16. layers.X.attn.q_b_proj.* -> layers.X.attn.wq_b.*
17. layers.X.attn.compressor.gate_proj.* -> layers.X.attn.compressor.wgate.*
18. layers.X.attn.compressor.kv_proj.* -> layers.X.attn.compressor.wkv.*
19. layers.X.attn.compressor.position_bias -> layers.X.attn.compressor.ape
20. layers.X.attn.compressor.indexer.gate_proj.* -> layers.X.attn.indexer.compressor.wgate.*
21. layers.X.attn.compressor.indexer.kv_proj.* -> layers.X.attn.indexer.compressor.wkv.*
22. layers.X.attn.compressor.indexer.position_bias -> layers.X.attn.indexer.compressor.ape
23. layers.X.attn.compressor.indexer.q_b_proj.* -> layers.X.attn.indexer.wq_b.*
24. layers.X.attn.compressor.indexer.scorer.weights_proj.* -> layers.X.attn.indexer.weights_proj.*
25. layers.X.mlp.* -> layers.X.ffn.*
26. layers.X.ffn.experts.Y.gate_proj.* -> layers.X.ffn.experts.Y.w1.*
27. layers.X.ffn.experts.Y.up_proj.* -> layers.X.ffn.experts.Y.w3.*
28. layers.X.ffn.experts.Y.down_proj.* -> layers.X.ffn.experts.Y.w2.*
29. layers.X.ffn.shared_experts.gate_proj.* -> layers.X.ffn.shared_experts.w1.*
30. layers.X.ffn.shared_experts.up_proj.* -> layers.X.ffn.shared_experts.w3.*
31. layers.X.ffn.shared_experts.down_proj.* -> layers.X.ffn.shared_experts.w2.*
32. model.hc_head.hc_base -> hc_head_base (flatten structure)
33. model.hc_head.hc_fn -> hc_head_fn
34. model.hc_head.hc_scale -> hc_head_scale
35. layers.X.attn_hc.base -> layers.X.hc_attn_base (reorder)
36. layers.X.attn_hc.fn -> layers.X.hc_attn_fn
37. layers.X.attn_hc.scale -> layers.X.hc_attn_scale
38. layers.X.ffn_hc.base -> layers.X.hc_ffn_base (reorder)
39. layers.X.ffn_hc.fn -> layers.X.hc_ffn_fn
40. layers.X.ffn_hc.scale -> layers.X.hc_ffn_scale

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
    # lm_head -> head
    if key.startswith("lm_head"):
        return key.replace("lm_head", "head")

    # model.embed_tokens.weight -> embed.weight
    if key == "model.embed_tokens.weight":
        return "embed.weight"

    # model.norm -> norm
    if key.startswith("model.norm"):
        return key.replace("model.norm", "norm")

    # model.hc_head.hc_base -> hc_head_base (flatten the structure)
    if key.startswith("model.hc_head."):
        # model.hc_head.hc_base -> hc_head_base
        # model.hc_head.hc_fn -> hc_head_fn
        # model.hc_head.hc_scale -> hc_head_scale
        return key.replace("model.hc_head.hc_", "hc_head_")

    # Handle layers (strip 'model.' prefix if present)
    if key.startswith("model.layers."):
        key = key[6:]  # Remove "model."

    # Process all layer keys (whether they started with model.layers or just layers)
    if key.startswith("layers."):
        # input_layernorm -> attn_norm
        if ".input_layernorm." in key:
            key = key.replace(".input_layernorm.", ".attn_norm.")

        # post_attention_layernorm -> ffn_norm
        if ".post_attention_layernorm." in key:
            key = key.replace(".post_attention_layernorm.", ".ffn_norm.")

        # self_attn -> attn
        if ".self_attn." in key:
            key = key.replace(".self_attn.", ".attn.")

        # After self_attn->attn conversion, fix nested compressor/indexer structure
        # compressor.indexer.kv_norm -> indexer.compressor.norm
        if ".attn.compressor.indexer.kv_norm." in key:
            key = key.replace(".attn.compressor.indexer.kv_norm.", ".attn.indexer.compressor.norm.")
        # compressor.kv_norm -> compressor.norm (only if not already handled above)
        elif ".attn.compressor.kv_norm." in key:
            key = key.replace(".attn.compressor.kv_norm.", ".attn.compressor.norm.")

        # q_a_norm -> q_norm
        if ".q_a_norm." in key:
            key = key.replace(".q_a_norm.", ".q_norm.")

        # attn_hc.base -> hc_attn_base (reorder)
        # layers.0.attn_hc.base -> layers.0.hc_attn_base
        if ".attn_hc." in key:
            key = key.replace(".attn_hc.", ".hc_attn_")

        # ffn_hc.base -> hc_ffn_base (reorder)
        # layers.0.ffn_hc.base -> layers.0.hc_ffn_base
        if ".ffn_hc." in key:
            key = key.replace(".ffn_hc.", ".hc_ffn_")

        # mlp -> ffn (all MLP references become FFN)
        if ".mlp." in key:
            key = key.replace(".mlp.", ".ffn.")

        # After mlp->ffn conversion, handle projections
        if ".ffn.experts." in key or ".ffn.shared_experts." in key:
            # gate_proj -> w1, up_proj -> w3, down_proj -> w2
            key = key.replace(".gate_proj.", ".w1.")
            key = key.replace(".up_proj.", ".w3.")
            key = key.replace(".down_proj.", ".w2.")

        # Attention projection name conversions (proj suffix -> w prefix)
        # Must be done carefully to avoid conflicts - process in specific order

        # First handle nested indexer compressor paths (most specific first)
        if ".attn.compressor.indexer.scorer.weights_proj" in key:
            key = key.replace(".attn.compressor.indexer.scorer.weights_proj", ".attn.indexer.weights_proj")
        if ".attn.compressor.indexer.gate_proj" in key:
            key = key.replace(".attn.compressor.indexer.gate_proj", ".attn.indexer.compressor.wgate")
        if ".attn.compressor.indexer.kv_proj" in key:
            key = key.replace(".attn.compressor.indexer.kv_proj", ".attn.indexer.compressor.wkv")
        if ".attn.compressor.indexer.position_bias" in key:
            key = key.replace(".attn.compressor.indexer.position_bias", ".attn.indexer.compressor.ape")
        if ".attn.compressor.indexer.q_b_proj" in key:
            key = key.replace(".attn.compressor.indexer.q_b_proj", ".attn.indexer.wq_b")

        # Then handle compressor paths
        if ".attn.compressor.gate_proj" in key:
            key = key.replace(".attn.compressor.gate_proj", ".attn.compressor.wgate")
        if ".attn.compressor.kv_proj" in key:
            key = key.replace(".attn.compressor.kv_proj", ".attn.compressor.wkv")
        if ".attn.compressor.position_bias" in key:
            key = key.replace(".attn.compressor.position_bias", ".attn.compressor.ape")

        # Finally handle top-level attention projections
        if ".attn.sinks" in key:
            key = key.replace(".attn.sinks", ".attn.attn_sink")
        if ".attn.kv_proj" in key:
            key = key.replace(".attn.kv_proj", ".attn.wkv")
        if ".attn.o_a_proj" in key:
            key = key.replace(".attn.o_a_proj", ".attn.wo_a")
        if ".attn.o_b_proj" in key:
            key = key.replace(".attn.o_b_proj", ".attn.wo_b")
        if ".attn.q_a_proj" in key:
            key = key.replace(".attn.q_a_proj", ".attn.wq_a")
        if ".attn.q_b_proj" in key:
            key = key.replace(".attn.q_b_proj", ".attn.wq_b")

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
