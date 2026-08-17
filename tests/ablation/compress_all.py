"""NVFP4 observer comparison: compress models with multiple observer configs.

All weights use NVFP4 E2M1 with FP8 E4M3 group scales, group_size=16.
Data-free (RTN) quantization — no calibration data needed.

Usage:
    torchrun --nproc_per_node=1 compress_all.py
    torchrun --nproc_per_node=1 compress_all.py --models meta-llama/Meta-Llama-3-8B
    torchrun --nproc_per_node=1 compress_all.py --configs fouroversix nvfp4_expanded_mse
    torchrun --nproc_per_node=1 compress_all.py --force
    torchrun --nproc_per_node=1 compress_all.py --list
"""

import argparse
import sys
from pathlib import Path

import torch
from compressed_tensors.offload import init_dist
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
from llmcompressor.utils import load_context

# ── Constants ────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
]

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

# ── Observer configs ─────────────────────────────────────────────────

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
            "expand": 1.5,
            "grid": 3,
            "maxshrink": 0.67,
            "norm": 2.0,
            "patience": 100000,
        },
    ),
    "nvfp4_expanded_mse": QuantizationArgs(
        **COMMON,
        observer="nvfp4_expanded_mse",
    ),
    "expand-3.4": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 3.4,
            "maxshrink": round(1 - 0.8 / 3.4, 4),
            "grid": 200.0,
            "patience": 200,
        },
    ),
    "expanded-norm1.8": QuantizationArgs(
        **COMMON,
        observer="nvfp4_expanded_mse",
        observer_kwargs={"norm": 1.8},
    ),
    "expanded-norm2.0": QuantizationArgs(
        **COMMON,
        observer="nvfp4_expanded_mse",
        observer_kwargs={"norm": 2.0},
    ),
    "expanded-norm2.2": QuantizationArgs(
        **COMMON,
        observer="nvfp4_expanded_mse",
        observer_kwargs={"norm": 2.2},
    ),
    "expanded-norm2.4": QuantizationArgs(
        **COMMON,
        observer="nvfp4_expanded_mse",
        observer_kwargs={"norm": 2.4},
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
    # ── Experiment A: no FP8 scale rounding ────────────────────────────
    "nofp8-1x-1.5x": QuantizationArgs(
        **{**COMMON, "scale_dtype": None},
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.5,
            "grid": 3,
            "maxshrink": 0.67,
            "norm": 2.0,
            "patience": 100000,
        },
    ),
    "nofp8-expanded": QuantizationArgs(
        **{**COMMON, "scale_dtype": None},
        observer="nvfp4_expanded_mse",
    ),
    # ── Experiment B: gs-prior with varying expand range ───────────────
    "gs-prior-1.0": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.0,
            "maxshrink": 0.20,
            "grid": 200,
            "norm": 2.4,
            "patience": 1000,
            "gs_prior": {"scale": 1.0, "fuse": True, "use_as_final": False},
        },
    ),
    "gs-prior-1.25": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.25,
            "maxshrink": round(1 - 0.8 / 1.25, 4),
            "grid": 200,
            "norm": 2.4,
            "patience": 1000,
            "gs_prior": {"scale": 1.25, "fuse": True, "use_as_final": False},
        },
    ),
    "gs-prior-1.5": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.5,
            "maxshrink": round(1 - 0.8 / 1.5, 4),
            "grid": 200,
            "norm": 2.4,
            "patience": 1000,
            "gs_prior": {"scale": 1.5, "fuse": True, "use_as_final": False},
        },
    ),
    "gs-prior-1.75": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.75,
            "maxshrink": round(1 - 0.8 / 1.75, 4),
            "grid": 200,
            "norm": 2.4,
            "patience": 1000,
            "gs_prior": {"scale": 1.75, "fuse": True, "use_as_final": False},
        },
    ),
    # ── Experiment C: gs-prior fuse/final variations ─────────────────
    "1x1.5x-gsp-local": QuantizationArgs(
        **COMMON,
        observer="memoryless_mse",
        observer_kwargs={
            "expand": 1.5,
            "grid": 3,
            "maxshrink": 0.67,
            "norm": 2.0,
            "patience": 100000,
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
            "expand": 1.5,
            "maxshrink": round(1 - 0.8 / 1.5, 4),
            "grid": 200,
            "norm": 2.4,
            "patience": 1000,
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

CONFIG_DESCRIPTIONS = {
    "fouroversix": "FourOverSix: per-block M=6/M=4 adaptive scaling, gs_max=256",
    "mse-1x-1.5x": "MSE 1x+1.5x: 2-point search matching FourOverSix search space",
    "nvfp4_expanded_mse": "NVFP4 Expanded MSE: 1.8x→0.8x range, 112 steps (default norm=2.4)",
    "expand-3.4": "Original ablation: expand=3.4, maxshrink=0.7647, grid=200, patience=200",
    "expanded-norm1.8": "NVFP4 Expanded MSE: norm=1.8",
    "expanded-norm2.0": "NVFP4 Expanded MSE: norm=2.0",
    "expanded-norm2.2": "NVFP4 Expanded MSE: norm=2.2",
    "expanded-norm2.4": "NVFP4 Expanded MSE: norm=2.4 (explicit)",
    "default-mse": "Default MSE: expand=1.0, maxshrink=0.20, grid=100, norm=2.4",
    "minmax": "MinMax: no MSE search, simple min/max scaling",
    "expanded-gs-prior": "NVFP4 Expanded MSE: with global_scale prior in search",
    "nofp8-1x-1.5x": "Exp A: mse-1x-1.5x WITHOUT FP8 scale rounding",
    "nofp8-expanded": "Exp A: expanded MSE WITHOUT FP8 scale rounding",
    "gs-prior-1.0": "Exp B: gs-prior, expand=1.0 (1.0x→0.8x, 40 steps)",
    "gs-prior-1.25": "Exp B: gs-prior, expand=1.25 (1.25x→0.8x, 72 steps)",
    "gs-prior-1.5": "Exp B: gs-prior, expand=1.5 (1.5x→0.8x, 94 steps)",
    "gs-prior-1.75": "Exp B: gs-prior, expand=1.75 (1.75x→0.8x, 109 steps)",
    "1x1.5x-gsp-local": "Exp C: 1x+1.5x search, gs_prior scale=1.8, local (no fusion), recalculate final GS",
    "expanded-gsp-1x-local": "Exp C: expanded 1.8x→0.8x, gs_prior scale=1.0, local, recalculate final GS",
    "expanded-gsp-fused-final": "Exp C: expanded 1.8x→0.8x, gs_prior scale=1.8, fused, use as final GS",
    "expand1.5-gsp-fused-final": "Exp C: expand 1.5x→0.8x, gs_prior scale=1.5, fused, use as final GS",
    "twopass-gsp-1.0": "Exp D: two-pass observer, gs_prior scale=1.0 in pass 1",
    "twopass-gsp-1.25": "Exp D: two-pass observer, gs_prior scale=1.25 in pass 1",
    "twopass-gsp-1.5": "Exp D: two-pass observer, gs_prior scale=1.5 in pass 1",
    "twopass-gsp-1.75": "Exp D: two-pass observer, gs_prior scale=1.75 in pass 1",
}


# ── Helpers ──────────────────────────────────────────────────────────


def model_short_name(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def compress(
    model_id: str,
    config_key: str,
    output_base: Path,
    force: bool = False,
):
    name = model_short_name(model_id)
    out = output_base / f"{name}-{config_key}"

    if out.exists() and not force:
        print(f"  SKIP (exists): {out}")
        return out

    print(f"\n{'='*70}")
    print(f"  {out}")
    print(f"  model:  {model_id}")
    print(f"  config: {config_key} — {CONFIG_DESCRIPTIONS.get(config_key, '')}")
    print(f"{'='*70}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    with load_context():
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto_offload"
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

    oneshot(
        model=model,
        recipe=recipe,
        output_dir=str(out),
    )
    tokenizer.save_pretrained(out)
    del model
    torch.cuda.empty_cache()
    print(f"  DONE: {out}\n")
    return out


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compress models with NVFP4 observer configs"
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="HuggingFace model IDs to compress",
    )
    parser.add_argument(
        "--configs", nargs="+", default=list(CONFIGS.keys()),
        help="Observer configs to run",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_BASE),
        help="Base directory for compressed models",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompress even if output exists",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List configs and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available configs:")
        for key, desc in CONFIG_DESCRIPTIONS.items():
            print(f"  {key:<25s}  {desc}")
        print(f"\nDefault models: {', '.join(DEFAULT_MODELS)}")
        sys.exit(0)

    for key in args.configs:
        if key not in CONFIGS:
            print(f"Unknown config: {key}")
            print(f"Available: {', '.join(CONFIGS.keys())}")
            sys.exit(1)

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    init_dist()

    total = len(args.models) * len(args.configs)
    idx = 0
    for model_id in args.models:
        for config_key in args.configs:
            idx += 1
            print(f"\n[{idx}/{total}] {model_short_name(model_id)} / {config_key}")
            compress(model_id, config_key, output_base, args.force)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
