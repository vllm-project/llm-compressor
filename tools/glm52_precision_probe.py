# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Read selected GLM checkpoint headers for L4 planning; never load weight payloads.

Standard-library-only. This is a layout preflight, NOT a numerical L4 pass.
Only the explicitly supplied existing local model directory is inspected.
"""

import argparse
import importlib.metadata
import json
import struct
from collections import Counter
from pathlib import Path

DIMENSIONS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "intermediate_size",
    "moe_intermediate_size",
    "n_routed_experts",
    "n_shared_experts",
    "first_k_dense_replace",
    "index_head_dim",
    "index_n_heads",
    "indexer_types",
    "num_nextn_predict_layers",
    "tie_word_embeddings",
)


def local_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or out-of-directory checkpoint file: {name}")
    return path


def read_header(path):
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Truncated safetensors header: {path.name}")
        size = struct.unpack("<Q", size_bytes)[0]
        if size > 100_000_000 or size > path.stat().st_size - 8:
            raise ValueError(f"Invalid safetensors header size: {path.name}")
        return json.loads(stream.read(size))


def inspect_checkpoint(model_dir, layers=None):
    root = model_dir.resolve(strict=True)
    config = json.loads(local_file(root, "config.json").read_text(encoding="utf-8"))
    count = config.get("num_hidden_layers")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("config.json must specify positive num_hidden_layers")
    indexers = config.get("indexer_types") or []
    if layers is None:
        layers = [0]
        # Cover a second indexer topology when explicitly described by config.
        if indexers:
            different = next(
                (i for i, value in enumerate(indexers[:count]) if value != indexers[0]),
                None,
            )
            if different is not None:
                layers.append(different)
        if len(layers) == 1 and count > 1:
            layers.append(1)
    layers = sorted(set(layers))
    if not layers or len(layers) > 4 or any(i < 0 or i >= count for i in layers):
        raise ValueError("Select one to four valid decoder layer indices")
    headers = {}
    if (root / "model.safetensors.index.json").is_file():
        index = json.loads(local_file(root, "model.safetensors.index.json").read_text())
        weight_map = index["weight_map"]
    else:
        header = read_header(local_file(root, "model.safetensors"))
        headers["model.safetensors"] = header
        weight_map = {
            name: "model.safetensors" for name in header if name != "__metadata__"
        }

    def selected(name):
        if name in {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}:
            return True
        for layer in layers:
            prefix = f"model.layers.{layer}."
            if name.startswith(prefix):
                tail = name[len(prefix) :]
                # One routed expert is sufficient for topology inspection. Packed
                # 3D expert tensors remain visible without reading their payload.
                if tail.startswith("mlp.experts."):
                    expert = tail.split(".")[2]
                    return not expert.isdigit() or expert == "0"
                return True
        return False

    tensors = {}
    errors, notes = [], []
    for name, shard in weight_map.items():
        if not selected(name):
            continue
        if shard not in headers:
            headers[shard] = read_header(local_file(root, shard))
        entry = headers[shard].get(name)
        if entry is None:
            errors.append(f"Index entry missing in shard header: {name}")
            continue
        tensors[name] = {"shape": entry["shape"], "dtype": entry["dtype"]}

    def expect(name, shape):
        item = tensors.get(name)
        if item is None:
            errors.append(f"Missing tensor: {name}")
        elif item["shape"] != shape:
            errors.append(f"Shape mismatch {name}: {item['shape']} != {shape}")

    required = [
        "hidden_size",
        "num_attention_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    ]
    valid_dimensions = all(
        isinstance(config.get(key), int)
        and not isinstance(config[key], bool)
        and config[key] > 0
        for key in required
    )
    if not valid_dimensions:
        errors.append("Missing or invalid positive MLA dimensions in config")
    else:
        h, heads, q, kv, key, rope, value = (config[key] for key in required)
        for dimension in (h, q, kv, value):
            if dimension % 32:
                errors.append(
                    f"QuaRot block_size=32 does not divide dimension {dimension}"
                )
        if config.get("num_key_value_heads", heads) != heads:
            errors.append("Current OV adapter requires expanded V heads")
        for layer in layers:
            prefix = f"model.layers.{layer}."
            shapes = {
                "input_layernorm.weight": [h],
                "post_attention_layernorm.weight": [h],
                "self_attn.q_a_layernorm.weight": [q],
                "self_attn.kv_a_layernorm.weight": [kv],
                "self_attn.q_a_proj.weight": [q, h],
                "self_attn.q_b_proj.weight": [heads * (key + rope), q],
                "self_attn.kv_a_proj_with_mqa.weight": [kv + rope, h],
                "self_attn.kv_b_proj.weight": [heads * (key + value), kv],
                "self_attn.o_proj.weight": [h, heads * value],
            }
            for tail, shape in shapes.items():
                expect(prefix + tail, shape)
        expect("model.norm.weight", [h])
    if config.get("quantization_config"):
        errors.append(
            "Checkpoint declares quantization; use the existing BF16 baseline"
        )
    for name, entry in tensors.items():
        if name.endswith(".weight") and entry["dtype"] not in {
            "BF16",
            "F16",
            "F32",
            "F64",
        }:
            errors.append(f"Non-floating-baseline weight: {name} ({entry['dtype']})")
        if ".experts." in name and len(entry["shape"]) == 3:
            notes.append(f"Packed experts require LLMC linearization: {name}")
    notes.append(
        "Only selected MLA shapes checked; indexer/expert/alias topology needs review"
    )
    if config.get("num_nextn_predict_layers", 0):
        notes.append(
            "Config declares MTP; exclude MTP before using the current modifiers"
        )
    versions = {}
    for package in ("torch", "transformers", "compressed-tensors", "llmcompressor"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema_version": 1,
        "scope": "selected_header_preflight_only",
        "numerical_l4_passed": False,
        "weight_payloads_read": False,
        "selected_shape_checks_passed": not errors,
        "selected_layers": layers,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "config": {key: config.get(key) for key in DIMENSIONS},
        "versions": versions,
        "selected_tensors": tensors,
        "dtype_counts": dict(Counter(item["dtype"] for item in tensors.values())),
        "errors": errors,
        "notes": notes,
        "next_required": (
            "Review headers, then validate transforms using selected weights "
            "and real activation caches; do not run L5 yet"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+")
    args = parser.parse_args()
    # Protect the supplied checkpoint from accidental overwrite by --output.
    if args.output.resolve().is_relative_to(args.model_dir.resolve()):
        parser.error("--output must be outside the model directory")
    try:
        report = inspect_checkpoint(args.model_dir, args.layers)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {
            "scope": "selected_header_preflight_only",
            "numerical_l4_passed": False,
            "selected_shape_checks_passed": False,
            "errors": [str(error)],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    passed = report["selected_shape_checks_passed"]
    print(
        f"Selected shape checks: {'PASS' if passed else 'NEEDS REVIEW'}; "
        f"report: {args.output}"
    )
    print(
        "This does not establish numerical L4 correctness. Return the JSON for review."
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
