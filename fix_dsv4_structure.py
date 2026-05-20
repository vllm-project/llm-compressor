"""
Convert NVFP4-FP8-BLOCK checkpoint format to Transformers-compatible format.

This script handles:
1. Name mapping (model. prefix removal, module renaming)
2. Dtype conversions (BF16 -> F32 for specific tensors)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def build_name_mapping(source_keys: list[str]) -> Dict[str, str]:
    """Build name mapping from source to target format."""
    mapping = {}

    for key in source_keys:
        new_key = key

        # Remove 'model.' prefix
        if new_key.startswith("model."):
            new_key = new_key[6:]

        # Rename modules
        new_key = new_key.replace("embed_tokens", "embed")
        new_key = new_key.replace("self_attn", "attn")
        new_key = new_key.replace(".mlp.", ".ffn.")
        new_key = new_key.replace("input_layernorm", "attn_norm")
        new_key = new_key.replace("post_attention_layernorm", "ffn_norm")

        # Flatten HC (head compression) tensors
        new_key = re.sub(r"hc_head\.hc_(base|fn|scale)", r"hc_head_\1", new_key)
        new_key = re.sub(
            r"layers\.(\d+)\.attn_hc\.(base|fn|scale)", r"layers.\1.hc_attn_\2", new_key
        )
        new_key = re.sub(
            r"layers\.(\d+)\.ffn_hc\.(base|fn|scale)", r"layers.\1.hc_ffn_\2", new_key
        )

        # Rename attention projections (but not in compressor.indexer)
        if ".compressor.indexer." not in new_key:
            new_key = new_key.replace("q_a_proj", "wq_a")
            new_key = new_key.replace("q_b_proj", "wq_b")
            new_key = new_key.replace("q_a_norm", "q_norm")
            new_key = new_key.replace("o_a_proj", "wo_a")
            new_key = new_key.replace("o_b_proj", "wo_b")
            # kv_proj -> wkv only when not in compressor
            if ".compressor." not in new_key:
                new_key = re.sub(r"\.kv_proj\.weight$", ".wkv.weight", new_key)

        new_key = new_key.replace("sinks", "attn_sink")

        # Rename shared expert projections
        if "shared_experts" in new_key:
            new_key = new_key.replace("gate_proj", "w1")
            new_key = new_key.replace("down_proj", "w2")
            new_key = new_key.replace("up_proj", "w3")

        mapping[key] = new_key

    return mapping


def should_convert_to_f32(key: str) -> bool:
    """Check if a tensor should be converted to F32."""
    f32_patterns = [
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "position_bias",
        "attn_sink",
        "e_score_correction_bias",
    ]
    return any(pattern in key for pattern in f32_patterns)


def convert_checkpoint(
    source_dir: Path,
    target_dir: Path,
    save_name_mapping: bool = True,
):
    """
    Convert checkpoint from NVFP4-FP8-BLOCK format to Transformers format.

    Args:
        source_dir: Source checkpoint directory
        target_dir: Target output directory
        save_name_mapping: Whether to save name mapping JSON
    """
    source_file = source_dir / "model.safetensors"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading source checkpoint from {source_file}")

    # Load all tensors
    tensors = {}
    with safe_open(str(source_file), framework="pt") as f:
        source_keys = list(f.keys())
        for key in tqdm(source_keys, desc="Loading tensors"):
            tensors[key] = f.get_tensor(key)

    print(f"Loaded {len(tensors)} tensors")

    # Build name mapping
    name_mapping = build_name_mapping(source_keys)

    # Convert tensors
    converted = {}

    for source_key in tqdm(source_keys, desc="Converting tensors"):
        target_key = name_mapping[source_key]
        tensor = tensors[source_key]

        # Convert dtype if needed
        if should_convert_to_f32(target_key) and tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)

        converted[target_key] = tensor

    print(f"\nConverted {len(converted)} tensors")

    # Save converted checkpoint
    target_file = target_dir / "model.safetensors"
    print(f"Saving converted checkpoint to {target_file}")
    save_file(converted, target_file)

    # Save name mapping if requested
    if save_name_mapping:
        mapping_file = target_dir / "name_mapping.json"
        with open(mapping_file, "w") as f:
            json.dump(name_mapping, f, indent=2)
        print(f"Saved name mapping to {mapping_file}")

    # Copy other files
    for file in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        source_path = source_dir / file
        if source_path.exists():
            target_path = target_dir / file
            target_path.write_bytes(source_path.read_bytes())
            print(f"Copied {file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NVFP4-FP8-BLOCK checkpoint to Transformers format"
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Source checkpoint directory (NVFP4-FP8-BLOCK format)",
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Target output directory (Transformers format)",
    )
    parser.add_argument(
        "--no-save-mapping",
        action="store_true",
        help="Don't save name mapping JSON",
    )

    args = parser.parse_args()

    convert_checkpoint(
        args.source_dir,
        args.target_dir,
        save_name_mapping=not args.no_save_mapping,
    )

    print("\n✓ Conversion complete!")


if __name__ == "__main__":
    main()
