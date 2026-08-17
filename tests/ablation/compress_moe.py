"""Compress MoE models without the DDP offload mechanism.

The standard compress_all.py uses init_dist() + auto_offload which creates
shared-memory files for each parameter. MoE models with hundreds of experts
exhaust vm.max_map_count (default 65530). This script uses device_map="auto"
instead.

Usage:
    python compress_moe.py --model Qwen/Qwen3-30B-A3B --configs fouroversix mse-1x-1.5x
"""

import argparse
import sys
from pathlib import Path

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.quant_args import FP8_E4M3_DATA
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

OUTPUT_BASE = Path("compressed-models")

COMMON = dict(
    num_bits=4,
    type=QuantizationType.FLOAT,
    strategy=QuantizationStrategy.TENSOR_GROUP,
    symmetric=True,
    dynamic=False,
    group_size=16,
    scale_dtype=FP8_E4M3_DATA.dtype,
    zp_dtype=FP8_E4M3_DATA.dtype,
)

CONFIGS = {
    "fouroversix": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.5,
            "grid": 3,
            "maxshrink": 0.67,
            "norm": 2.0,
            "patience": 100000,
            "gs_prior": {"scale": 448 / 256, "fuse": True, "use_as_final": True},
        },
    ),
    "mse-1x-1.5x": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.5, "grid": 3, "maxshrink": 0.67,
            "norm": 2.0, "patience": 100000,
        },
    ),
    "nvfp4_expanded_mse": QuantizationArgs(
        **COMMON,
        observer="nvfp4_expanded_mse",
    ),
    "default-mse": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
    ),
    "minmax": QuantizationArgs(
        **COMMON,
        observer="memoryless_minmax",
    ),
    "expanded-gs-prior": QuantizationArgs(
        **COMMON,
        observer="nvfp4_expanded_mse",
        observer_kwargs={
            "gs_prior": {"scale": 1.8, "fuse": True, "use_as_final": False},
        },
    ),
    "nofp8-1x-1.5x": QuantizationArgs(
        **{**COMMON, "scale_dtype": None},
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.5, "grid": 3, "maxshrink": 0.67,
            "norm": 2.0, "patience": 100000,
        },
    ),
    "nofp8-expanded": QuantizationArgs(
        **{**COMMON, "scale_dtype": None},
        observer="nvfp4_expanded_mse",
    ),
    "gs-prior-1.0": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.0, "maxshrink": 0.20, "grid": 200,
            "norm": 2.4, "patience": 1000,
            "gs_prior": {"scale": 1.0, "fuse": True, "use_as_final": False},
        },
    ),
    "gs-prior-1.25": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.25, "maxshrink": round(1 - 0.8 / 1.25, 4), "grid": 200,
            "norm": 2.4, "patience": 1000,
            "gs_prior": {"scale": 1.25, "fuse": True, "use_as_final": False},
        },
    ),
    "gs-prior-1.5": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.5, "maxshrink": round(1 - 0.8 / 1.5, 4), "grid": 200,
            "norm": 2.4, "patience": 1000,
            "gs_prior": {"scale": 1.5, "fuse": True, "use_as_final": False},
        },
    ),
    "gs-prior-1.75": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.75, "maxshrink": round(1 - 0.8 / 1.75, 4), "grid": 200,
            "norm": 2.4, "patience": 1000,
            "gs_prior": {"scale": 1.75, "fuse": True, "use_as_final": False},
        },
    ),
    # ── Experiment C: gs-prior fuse/final variations ─────────────────
    "1x1.5x-gsp-local": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.5, "grid": 3, "maxshrink": 0.67,
            "norm": 2.0, "patience": 100000,
            "gs_prior": {"scale": 1.8, "fuse": False, "use_as_final": False},
        },
    ),
    "expanded-gsp-1x-local": QuantizationArgs(
        **COMMON,
        observer="nvfp4_expanded_mse",
        observer_kwargs={
            "gs_prior": {"scale": 1.0, "fuse": False, "use_as_final": False},
        },
    ),
    "expanded-gsp-fused-final": QuantizationArgs(
        **COMMON,
        observer="nvfp4_expanded_mse",
        observer_kwargs={
            "gs_prior": {"scale": 1.8, "fuse": True, "use_as_final": True},
        },
    ),
    "expand1.5-gsp-fused-final": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.5, "maxshrink": round(1 - 0.8 / 1.5, 4), "grid": 200,
            "norm": 2.4, "patience": 1000,
            "gs_prior": {"scale": 1.5, "fuse": True, "use_as_final": True},
        },
    ),
    # ── Experiment D: two-pass observer with gs_prior ─────────────────
    "twopass-gsp-1.0": QuantizationArgs(
        **COMMON,
        observer="nvfp4_twopass",
        observer_kwargs={
            "gs_prior": {"scale": 1.0, "fuse": True},
        },
    ),
    "twopass-gsp-1.25": QuantizationArgs(
        **COMMON,
        observer="nvfp4_twopass",
        observer_kwargs={
            "gs_prior": {"scale": 1.25, "fuse": True},
        },
    ),
    "twopass-gsp-1.5": QuantizationArgs(
        **COMMON,
        observer="nvfp4_twopass",
        observer_kwargs={
            "gs_prior": {"scale": 1.5, "fuse": True},
        },
    ),
    "twopass-gsp-1.75": QuantizationArgs(
        **COMMON,
        observer="nvfp4_twopass",
        observer_kwargs={
            "gs_prior": {"scale": 1.75, "fuse": True},
        },
    ),
}


def model_short_name(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def compress(model_id, config_key, output_base, force=False):
    name = model_short_name(model_id)
    out = output_base / f"{name}-{config_key}"

    if out.exists() and not force:
        print(f"  SKIP (exists): {out}")
        return out

    print(f"\n{'='*70}")
    print(f"  {name} / {config_key}")
    print(f"{'='*70}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto"
    )

    recipe = QuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=CONFIGS[config_key],
            )
        },
        ignore=["lm_head"],
    )

    oneshot(model=model, recipe=recipe, output_dir=str(out))
    tokenizer.save_pretrained(out)
    del model
    torch.cuda.empty_cache()
    print(f"  DONE: {out}\n")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_BASE))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    total = len(args.configs)
    for i, config_key in enumerate(args.configs, 1):
        if config_key not in CONFIGS:
            print(f"Unknown config: {config_key}")
            sys.exit(1)
        print(f"\n[{i}/{total}] {model_short_name(args.model)} / {config_key}")
        compress(args.model, config_key, output_base, args.force)


if __name__ == "__main__":
    main()
