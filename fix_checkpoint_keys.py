import argparse
import json
import os
import re
import shutil
import struct


def rename_key(key: str) -> str:
    if not key.startswith("model."):
        return key

    key = key[len("model."):]

    # Indexer compressor parts (must come before general compressor/attn renames)
    # compressor.indexer.{ape,wgate,wkv} → indexer.compressor.{ape,wgate,wkv}
    key = re.sub(
        r"\.compressor\.indexer\.(ape|wgate|wkv)", r".indexer.compressor.\1", key
    )
    # compressor.indexer.kv_norm → indexer.compressor.norm
    key = key.replace(".compressor.indexer.kv_norm.", ".indexer.compressor.norm.")
    # compressor.indexer.{weights_proj,wq_b} → indexer.{weights_proj,wq_b}
    key = re.sub(
        r"\.compressor\.indexer\.(weights_proj|wq_b)", r".indexer.\1", key
    )

    # Compressor norm: compressor.kv_norm → compressor.norm
    key = key.replace(".compressor.kv_norm.", ".compressor.norm.")

    # Attention: self_attn → attn
    key = key.replace(".self_attn.", ".attn.")

    # MLP → FFN
    key = key.replace(".mlp.", ".ffn.")

    # Layer norms
    key = key.replace(".input_layernorm.", ".attn_norm.")
    key = key.replace(".post_attention_layernorm.", ".ffn_norm.")

    # Shared expert projections
    key = key.replace(".shared_experts.gate_proj.", ".shared_experts.w1.")
    key = key.replace(".shared_experts.up_proj.", ".shared_experts.w3.")
    key = key.replace(".shared_experts.down_proj.", ".shared_experts.w2.")

    return key


def process_safetensors_file(src_path: str, dst_path: str):
    with open(src_path, "rb") as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_json = f.read(header_size)
        data_start = 8 + header_size

    header = json.loads(header_json)

    new_header = {}
    for key, value in header.items():
        if key == "__metadata__":
            new_header[key] = value
        else:
            new_header[rename_key(key)] = value

    new_header_json = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    new_header_size_bytes = struct.pack("<Q", len(new_header_json))

    chunk_size = 64 * 1024 * 1024  # 64MB
    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        dst.write(new_header_size_bytes)
        dst.write(new_header_json)
        src.seek(data_start)
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)

    basename = os.path.basename(src_path)
    renamed = sum(1 for k in header if k != "__metadata__" and rename_key(k) != k)
    total = sum(1 for k in header if k != "__metadata__")
    print(f"  {basename}: renamed {renamed}/{total} keys")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.input_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)

    new_weight_map = {}
    for key, filename in index["weight_map"].items():
        new_weight_map[rename_key(key)] = filename
    index["weight_map"] = new_weight_map

    with open(os.path.join(args.output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print("Updated model.safetensors.index.json")

    safetensors_files = sorted(
        f for f in os.listdir(args.input_dir) if f.endswith(".safetensors")
    )
    for i, filename in enumerate(safetensors_files):
        src = os.path.join(args.input_dir, filename)
        dst = os.path.join(args.output_dir, filename)
        print(f"[{i+1}/{len(safetensors_files)}] Processing {filename}...")
        process_safetensors_file(src, dst)

    for filename in os.listdir(args.input_dir):
        if filename.endswith(".safetensors") or filename == "model.safetensors.index.json":
            continue
        src = os.path.join(args.input_dir, filename)
        dst = os.path.join(args.output_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"Copied {filename}")

    print("Done!")


if __name__ == "__main__":
    main()
