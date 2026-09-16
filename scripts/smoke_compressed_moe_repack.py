#!/usr/bin/env python3
"""
Resource-gated smoke for compressed 3D MoE repack (issue #3183).

Expected workflow:
  compress_model(model) -> repack_moe(model) -> save_pretrained(out)

Default model is the tiny Inkling checkpoint. Requires a transformers build
that ships Inkling (the 0.6B config was saved with 5.15.0.dev0). Pass
--model Qwen/Qwen3.6-35B-A3B for the large-model layout check; that download
is intentionally not part of CI.

Saved Inkling routed-expert suffixes must match
RedHatAI/Inkling-NVFP4-FP8-BLOCK:
  mlp.experts.w13_weight.{weight_packed,weight_scale,weight_global_scale,input_global_scale}
  mlp.experts.w2_weight.{...}
and must not contain per-expert ``experts.{i}`` keys.

Usage:
  python scripts/smoke_compressed_moe_repack.py --check-reference
  python scripts/smoke_compressed_moe_repack.py
  python scripts/smoke_compressed_moe_repack.py --scheme FP8
  python scripts/smoke_compressed_moe_repack.py --model Qwen/Qwen3.6-35B-A3B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INKLING_ID = "inference-optimization/Inkling-0.6B-A0.6B"
REFERENCE_ID = "RedHatAI/Inkling-NVFP4-FP8-BLOCK"
QWEN35B_ID = "Qwen/Qwen3.6-35B-A3B"
EXPECTED_INKLING_AFTER_EXPERTS = (
    "w13_weight.input_global_scale",
    "w13_weight.weight_global_scale",
    "w13_weight.weight_packed",
    "w13_weight.weight_scale",
    "w2_weight.input_global_scale",
    "w2_weight.weight_global_scale",
    "w2_weight.weight_packed",
    "w2_weight.weight_scale",
)


def _expert_keys(save_dir: Path) -> list[str]:
    from safetensors import safe_open

    keys: list[str] = []
    for path in save_dir.glob("*.safetensors"):
        with safe_open(path, framework="pt") as handle:
            keys.extend(k for k in handle.keys() if "mlp.experts" in k)
    return sorted(keys)


def _after_experts(key: str) -> str:
    parts = key.split(".")
    return ".".join(parts[parts.index("experts") + 1 :])


def check_reference_index() -> None:
    """Validate published Inkling NVFP4 keys without downloading weights."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REFERENCE_ID, "model.safetensors.index.json")
    with open(path) as handle:
        weight_map = json.load(handle)["weight_map"]
    expert_keys = [key for key in weight_map if "mlp.experts" in key]
    assert expert_keys, f"No mlp.experts keys in {REFERENCE_ID}"
    assert not any(".experts.0." in key for key in expert_keys)
    suffixes = {_after_experts(key) for key in expert_keys}
    assert suffixes == set(EXPECTED_INKLING_AFTER_EXPERTS), suffixes
    print(f"{REFERENCE_ID} expert suffixes OK:", sorted(suffixes))


def _model_cls_for(model_id: str):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_id)
    arch = (getattr(config, "architectures", None) or [""])[0]
    if "CausalLM" in arch:
        return AutoModelForCausalLM
    for name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
        try:
            module = __import__("transformers", fromlist=[name])
            return getattr(module, name)
        except AttributeError:
            continue
    return AutoModelForCausalLM


def run_smoke(args: argparse.Namespace) -> None:
    import torch
    from compressed_tensors import ModelCompressor
    from datasets import Dataset
    from transformers import AutoProcessor

    from llmcompressor import oneshot
    from llmcompressor.modeling.moe.linear_experts import (
        CompressedFusedLinear,
        LinearExperts2D,
    )
    from llmcompressor.modeling.moe.linearize import load_quantizable_moe, repack_moe
    from llmcompressor.modifiers.quantization import QuantizationModifier

    model_cls = _model_cls_for(args.model)
    print(f"Loading {args.model} with load_quantizable_moe({model_cls.__name__}) ...")
    with load_quantizable_moe(model_cls):
        model = model_cls.from_pretrained(
            args.model, device_map="auto", dtype=torch.bfloat16
        )

    experts = next(
        module
        for module in model.modules()
        if isinstance(module, LinearExperts2D)
    )
    print(f"Linearized experts: {type(experts).__name__}")

    texts = [f"calibration sample {i}" for i in range(args.num_samples)]
    ds = Dataset.from_dict({"text": texts})
    recipe = QuantizationModifier(
        targets="Linear",
        scheme=args.scheme,
        ignore=["lm_head", "re:.*visual.*", "re:.*mlp.gate$"],
    )
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_samples,
    )

    compressor = ModelCompressor.from_pretrained_model(model)
    print("Compressing quantized Linears ...")
    compressor.compress_model(model, skip_compressed=True)

    print("Calling repack_moe ...")
    repack_moe(model)
    fused = next(
        module
        for name, module in model.named_modules()
        if name.endswith("mlp.experts") and not isinstance(module, LinearExperts2D)
    )
    assert isinstance(getattr(fused, "gate_up_proj", None), CompressedFusedLinear)
    assert isinstance(getattr(fused, "down_proj", None), CompressedFusedLinear)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {args.out} ...")
    model.save_pretrained(args.out, safe_serialization=True)
    try:
        AutoProcessor.from_pretrained(args.model).save_pretrained(args.out)
    except Exception:
        pass

    expert_keys = _expert_keys(args.out)
    linearized = [key for key in expert_keys if ".experts.0." in key]
    assert not linearized, f"Found per-expert keys after repack: {linearized[:10]}"

    if args.model == INKLING_ID:
        after = {_after_experts(key) for key in expert_keys}
        missing = set(EXPECTED_INKLING_AFTER_EXPERTS) - after
        assert not missing, (
            f"Missing Inkling suffixes {missing} in {sorted(after)[:20]}"
        )
    else:
        packed = [
            key
            for key in expert_keys
            if "weight_packed" in key or "weight_scale" in key
        ]
        assert packed, f"No compressed 3D expert keys found: {expert_keys[:20]}"
        if args.model == QWEN35B_ID:
            assert any(
                "gate_up_proj" in key or key.endswith("down_proj.weight_packed")
                or "down_proj." in key
                for key in packed
            ), packed[:20]

    print("Packed expert keys (sample):", expert_keys[:12])
    print("SMOKE PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=INKLING_ID)
    parser.add_argument("--scheme", default="NVFP4", help="FP8 or NVFP4")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./compressed-moe-repack-smoke"),
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help="Only validate RedHatAI/Inkling-NVFP4-FP8-BLOCK expert key suffixes",
    )
    args = parser.parse_args()

    if args.check_reference:
        check_reference_index()
        return
    run_smoke(args)


if __name__ == "__main__":
    main()
