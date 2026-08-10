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

OUTPUT_BASE = Path("/tmp/fouroversix-sanity")

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
        observer="fouroversix",
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
        observer_kwargs={"use_global_scale_prior": True},
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
