"""Resave a DeepSeek-V4-Flash NVFP4 checkpoint with key names matching the BF16
checkpoint structure. Quantization parameter suffixes (weight_packed, weight_scale,
input_global_scale, weight_global_scale) are preserved; only prefixes and module
names are changed."""

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def rename_key(key: str) -> str:
    if key == "head.weight":
        return key

    if key.startswith("model."):
        key = key[len("model."):]

    top_level = {
        "embed_tokens.weight": "embed.weight",
        "norm.weight": "norm.weight",
        "hc_head.hc_base": "hc_head_base",
        "hc_head.hc_fn": "hc_head_fn",
        "hc_head.hc_scale": "hc_head_scale",
    }
    if key in top_level:
        return top_level[key]

    m = re.match(r"(layers\.\d+\.)(.*)", key)
    if not m:
        raise ValueError(f"Unrecognized key: {key}")

    prefix = m.group(1)
    rest = m.group(2)

    # --- layer norms ---
    if rest == "input_layernorm.weight":
        return prefix + "attn_norm.weight"
    if rest == "post_attention_layernorm.weight":
        return prefix + "ffn_norm.weight"

    # --- hardware counters ---
    hc_map = {
        "attn_hc.base": "hc_attn_base",
        "attn_hc.fn": "hc_attn_fn",
        "attn_hc.scale": "hc_attn_scale",
        "ffn_hc.base": "hc_ffn_base",
        "ffn_hc.fn": "hc_ffn_fn",
        "ffn_hc.scale": "hc_ffn_scale",
    }
    if rest in hc_map:
        return prefix + hc_map[rest]

    # --- compressor.indexer (most specific first) ---
    ci_exact = {
        "self_attn.compressor.indexer.gate_proj.weight": "attn.indexer.compressor.wgate.weight",
        "self_attn.compressor.indexer.kv_norm.weight": "attn.indexer.compressor.norm.weight",
        "self_attn.compressor.indexer.kv_proj.weight": "attn.indexer.compressor.wkv.weight",
        "self_attn.compressor.indexer.position_bias": "attn.indexer.compressor.ape",
        "self_attn.compressor.indexer.weights_proj.weight": "attn.indexer.weights_proj.weight",
    }
    if rest in ci_exact:
        return prefix + ci_exact[rest]
    m2 = re.match(r"self_attn\.compressor\.indexer\.q_b_proj\.(.*)", rest)
    if m2:
        return prefix + "attn.indexer.wq_b." + m2.group(1)

    # --- compressor (without indexer) ---
    c_exact = {
        "self_attn.compressor.gate_proj.weight": "attn.compressor.wgate.weight",
        "self_attn.compressor.kv_norm.weight": "attn.compressor.norm.weight",
        "self_attn.compressor.kv_proj.weight": "attn.compressor.wkv.weight",
        "self_attn.compressor.position_bias": "attn.compressor.ape",
    }
    if rest in c_exact:
        return prefix + c_exact[rest]

    # --- self-attention (exact matches) ---
    attn_exact = {
        "self_attn.sinks": "attn.attn_sink",
        "self_attn.kv_norm.weight": "attn.kv_norm.weight",
        "self_attn.q_a_norm.weight": "attn.q_norm.weight",
    }
    if rest in attn_exact:
        return prefix + attn_exact[rest]

    # --- self-attention projections (with possible quant suffixes) ---
    attn_proj_map = {
        "self_attn.kv_proj": "attn.wkv",
        "self_attn.o_a_proj": "attn.wo_a",
        "self_attn.o_b_proj": "attn.wo_b",
        "self_attn.q_a_proj": "attn.wq_a",
        "self_attn.q_b_proj": "attn.wq_b",
    }
    for old, new in attn_proj_map.items():
        m2 = re.match(rf"{re.escape(old)}\.(.*)", rest)
        if m2:
            return prefix + new + "." + m2.group(1)

    # --- MLP gate ---
    gate_map = {
        "mlp.gate.weight": "ffn.gate.weight",
        "mlp.gate.tid2eid": "ffn.gate.tid2eid",
        "mlp.gate.e_score_correction_bias": "ffn.gate.bias",
    }
    if rest in gate_map:
        return prefix + gate_map[rest]

    # --- MLP experts ---
    proj_map = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
    m2 = re.match(r"mlp\.experts\.(\d+)\.(gate_proj|down_proj|up_proj)\.(.*)", rest)
    if m2:
        eid, proj, suffix = m2.group(1), m2.group(2), m2.group(3)
        return prefix + f"ffn.experts.{eid}.{proj_map[proj]}.{suffix}"

    # --- MLP shared experts ---
    m2 = re.match(r"mlp\.shared_experts\.(gate_proj|down_proj|up_proj)\.(.*)", rest)
    if m2:
        proj, suffix = m2.group(1), m2.group(2)
        return prefix + f"ffn.shared_experts.{proj_map[proj]}.{suffix}"

    raise ValueError(f"Unrecognized key: layers.*.{rest}")


def main():
    parser = argparse.ArgumentParser(
        description="Resave NVFP4 checkpoint with BF16-style key names"
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    index_path = args.input_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    shard_names = sorted(set(index["weight_map"].values()))

    new_weight_map = {}
    for old_key, shard_name in index["weight_map"].items():
        new_weight_map[rename_key(old_key)] = shard_name

    for i, shard_name in enumerate(shard_names):
        src = args.input_dir / shard_name
        dst = args.output_dir / shard_name
        print(f"[{i + 1}/{len(shard_names)}] Processing {shard_name} ...")

        tensors = {}
        with safe_open(str(src), framework="pt") as f:
            for key in f.keys():
                tensors[rename_key(key)] = f.get_tensor(key)

        save_file(tensors, str(dst))
        del tensors
        print(f"  Saved {dst}")

    new_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    out_index = args.output_dir / "model.safetensors.index.json"
    with open(out_index, "w") as f:
        json.dump(new_index, f, indent=2, sort_keys=False)
    print(f"Saved {out_index}")

    for name in ("config.json", "generation_config.json",
                 "tokenizer.json", "tokenizer_config.json"):
        src = args.input_dir / name
        if src.exists():
            shutil.copy2(src, args.output_dir / name)
            print(f"Copied {name}")

    print("Done.")


if __name__ == "__main__":
    main()
