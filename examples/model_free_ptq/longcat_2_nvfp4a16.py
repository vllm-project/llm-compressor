"""
LongCat-2.0 NVFP4A16 Quantization Example

Quantizes LongCat-2.0 (1.6T total params, 48B active) to NVFP4A16 using model_free_ptq,
then patches the output model to include n-gram embedding weights and the correct config
for vLLM's ngram-aware model loader.

Usage:
   MODEL_ID=/path/to/LongCat-2.0 HF_HUB_DIR=/path/to/output python longcat_2_nvfp4a16.py
"""

import json
import os

import torch

os.environ["TORCH_COMPILE_DISABLE"] = "1"

from safetensors import safe_open
from safetensors.torch import save_file

from llmcompressor import model_free_ptq


def patch_for_ngram(source_model_path: str, save_dir: str) -> None:
    """Copy ngram embedding weights from the source model and patch config.

    The model_free_ptq flow quantizes linear layers but skips embeddings entirely,
    so the ngram embedding weights are not present in the output. This function:
    1. Copies the 32 ngram embedding weights (16 embedders + 16 projections) from
       the source model into a new safetensors shard in the quantized output.
    2. Patches config.json so vLLM loads the ngram-aware model class:
       - architectures -> ["LongcatCausalLM"]
       - removes model_type (triggers vLLM auto-detection)
       - ensures oe_vocab_size_ratio, oe_neighbor_num, oe_split_num are present
    """
    print("\n=== Patching for n-gram embeddings ===")

    # --- Step 1: Find and copy ngram weights ---
    source_index_path = os.path.join(source_model_path, "model.safetensors.index.json")
    if not os.path.exists(source_index_path):
        print(f"WARNING: No safetensors index at {source_index_path}, skipping ngram patch")
        return

    with open(source_index_path) as f:
        source_index = json.load(f)

    ngram_sources: dict[str, list[str]] = {}
    for weight_name, shard_file in source_index["weight_map"].items():
        if "ngram_embeddings" in weight_name and "mtp" not in weight_name:
            ngram_sources.setdefault(shard_file, []).append(weight_name)

    if not ngram_sources:
        print("No ngram_embeddings weights found in source model, skipping")
        return

    ngram_weights = {}
    for shard_file, weight_names in sorted(ngram_sources.items()):
        shard_path = os.path.join(source_model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in weight_names:
                ngram_weights[name] = f.get_tensor(name)
                shape = ngram_weights[name].shape
                dtype = ngram_weights[name].dtype
                print(f"  Copied {name}: {list(shape)} {dtype}")

    ngram_shard = "model-ngram.safetensors"
    save_file(ngram_weights, os.path.join(save_dir, ngram_shard))
    print(f"  Saved {len(ngram_weights)} ngram weights to {ngram_shard}")

    # --- Step 2: Update safetensors index ---
    output_index_path = os.path.join(save_dir, "model.safetensors.index.json")
    with open(output_index_path) as f:
        output_index = json.load(f)

    for weight_name in ngram_weights:
        output_index["weight_map"][weight_name] = ngram_shard

    with open(output_index_path, "w") as f:
        json.dump(output_index, f, indent=2)
    print(f"  Updated weight index with {len(ngram_weights)} entries")

    # --- Step 3: Patch config.json ---
    source_config_path = os.path.join(source_model_path, "config.json")
    output_config_path = os.path.join(save_dir, "config.json")

    with open(source_config_path) as f:
        source_config = json.load(f)
    with open(output_config_path) as f:
        output_config = json.load(f)

    output_config["architectures"] = ["LongcatCausalLM"]
    output_config.pop("model_type", None)

    ngram_fields = [
        "oe_vocab_size_ratio",
        "oe_neighbor_num",
        "oe_split_num",
        "moe_topk",
        "zero_expert_num",
        "zero_expert_type",
        "use_mla",
        "attention_method",
    ]
    for field in ngram_fields:
        if field in source_config and field not in output_config:
            output_config[field] = source_config[field]
            print(f"  Config: added {field} = {source_config[field]}")

    with open(output_config_path, "w") as f:
        json.dump(output_config, f, indent=2)

    print("=== N-gram patch complete ===\n")


# Source model - can be HuggingFace repo or local path
MODEL_ID = os.environ.get("MODEL_ID", "meituan-longcat/LongCat-2.0")

# Save quantized model
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~"))
HF_HUB_DIR = os.environ.get("HF_HUB_DIR", HF_HOME)
SAVE_DIR = os.path.join(HF_HUB_DIR, "LongCat-2.0-NVFP4A16")

print(f"Quantizing model from: {MODEL_ID}")
print(f"Saving quantized model to: {SAVE_DIR}")

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="NVFP4A16",
    ignore=[
        "model.embed_tokens",
        "lm_head",
        "re:.*gate$",
        "re:.*norm.*",
        "re:.*kv_a_proj_with_mqa$",
        "re:.*q_a_proj$",
        "re:.*mtp.*",
    ],
    max_workers=2,
)

# Resolve the source model path for ngram weight copying.
# MODEL_ID can be a local path or a HuggingFace repo ID.
source_path = MODEL_ID
if not os.path.isdir(source_path):
    from huggingface_hub import snapshot_download
    source_path = snapshot_download(MODEL_ID)

patch_for_ngram(source_path, SAVE_DIR)

print(f"Quantized model saved to: {SAVE_DIR}")
print()
print("To serve with vLLM:")
print(f"  vllm serve {SAVE_DIR} --trust-remote-code \\")
print("    --tensor-parallel-size 4 --max-model-len 512 --cpu-offload-gb 100 \\")
print("    --gpu-memory-utilization 0.99 --enforce-eager \\")
print('    --kernel-config \'{"moe_backend": "emulation"}\' \\')
print('    --reasoning-parser longcat --override-generation-config \'{"thinking": true}\'')
