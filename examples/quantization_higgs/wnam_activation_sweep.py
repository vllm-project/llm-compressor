"""
WNaM activation quantization sweep for HIGGS analysis.

Sweeps WNaM candidate scheme combinations with varying weight and activation
bitwidth budgets to understand how activation quantization affects
mixed-precision quality.

Available WNaM schemes (INT, group_size=128 weights, TOKEN dynamic activations):
  W2A4, W2A8, W2A16
  W3A4, W3A8, W3A16
  W4A4, W4A8, W4A16
  W5A8, W5A16
  W6A8, W6A16
  W7A8, W7A16
  W8A16

Usage:
    # Sweep weight-only schemes (baseline, no activation quant)
    python wnam_activation_sweep.py --model meta-llama/Meta-Llama-3-8B --short Llama-3-8B \\
        --schemes W4A16 W8A16 --weight-bits 4.5 5.0 5.5 6.0

    # Sweep with activation quantization (W4A8 + W8A8 candidates)
    python wnam_activation_sweep.py --model meta-llama/Meta-Llama-3-8B --short Llama-3-8B \\
        --schemes W4A8 W8A16 --weight-bits 5.0 6.0

    # Full WNaM sweep with both weight and activation budgets
    python wnam_activation_sweep.py --model meta-llama/Meta-Llama-3-8B --short Llama-3-8B \\
        --schemes W4A8 W4A16 W8A8 W8A16 --weight-bits 5.0 6.0 --act-bits 10.0 12.0

    # Config-only mode (skip quantization, just generate configs)
    python wnam_activation_sweep.py --model meta-llama/Meta-Llama-3-8B --short Llama-3-8B \\
        --schemes W4A8 W4A16 W8A16 --weight-bits 5.0 6.0 --config-only
"""

import argparse
import json
import os
import time

import torch

from llmcompressor.entrypoints.higgs import get_higgs_config


IGNORE = [
    "lm_head",
    "re:.*embed_tokens",
    "re:.*vision_tower.*",
    "re:.*audio_tower.*",
    "re:.*multi_modal_projector.*",
    "re:.*embed_audio.*",
    "re:.*embed_vision.*",
]


def save_dir(model_short, schemes_tag, wbits, abits):
    act_tag = f"-A{abits}avg" if abits is not None else ""
    return os.path.expanduser(
        f"~/hf_hub/{model_short}-HIGGS-{schemes_tag}-W{wbits}avg{act_tag}"
    )


def config_path(model_short, schemes_tag, wbits, abits):
    act_tag = f"-A{abits}avg" if abits is not None else ""
    return os.path.expanduser(
        f"~/hf_hub/{model_short}-HIGGS-{schemes_tag}-W{wbits}avg{act_tag}-config.json"
    )


def schemes_tag(schemes):
    return "+".join(sorted(schemes))


def save_config(config, path, schemes, wbits, abits):
    data = {
        "candidate_schemes": schemes,
        "target_avg_weight_bits": wbits,
        "target_avg_act_bits": abits,
        "config_groups": {},
    }
    for name, scheme in config.config_groups.items():
        data["config_groups"][name] = scheme.model_dump()

    # Compute scheme distribution
    scheme_counts = {}
    for name, scheme in config.config_groups.items():
        n_targets = len(scheme.targets)
        scheme_counts[name] = n_targets
    data["scheme_distribution"] = scheme_counts

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Config saved to {path}")


def run_sweep(model_id, model_short, schemes, weight_bits_list, act_bits_list,
              config_only=False):
    tag = schemes_tag(schemes)
    results = []

    for wbits in weight_bits_list:
        for abits in act_bits_list:
            out = save_dir(model_short, tag, wbits, abits)
            cfg_path = config_path(model_short, tag, wbits, abits)

            # Skip if already done
            if not config_only and os.path.exists(out) and any(
                f.endswith(".safetensors") for f in os.listdir(out)
            ):
                print(f"[SKIP] W{wbits}avg A{abits}avg already exists at {out}")
                continue
            if config_only and os.path.exists(cfg_path):
                print(f"[SKIP] Config already exists at {cfg_path}")
                continue

            act_label = f" act_budget={abits}" if abits is not None else ""
            print(f"\n{'='*80}")
            print(
                f"Schemes: {schemes} | weight_budget={wbits}{act_label}"
            )
            print(f"{'='*80}")

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start = time.time()
            try:
                config = get_higgs_config(
                    model_stub=model_id,
                    candidate_schemes=schemes,
                    targets="Linear",
                    ignore=IGNORE,
                    enforce_fused_layer_constraints=True,
                    target_avg_bitwidth=wbits,
                    target_avg_act_bitwidth=abits,
                    device="cuda:0",
                )
            except ValueError as e:
                print(f"  FAILED: {e}")
                results.append({
                    "schemes": schemes,
                    "weight_bits": wbits,
                    "act_bits": abits,
                    "status": f"FAILED: {e}",
                })
                continue

            elapsed = time.time() - start

            peak_gb = 0
            if torch.cuda.is_available():
                peak_gb = torch.cuda.max_memory_allocated() / 1024**3

            # Count scheme distribution
            scheme_counts = {}
            for gname, scheme in config.config_groups.items():
                n_targets = len(scheme.targets)
                scheme_counts[gname] = n_targets

            # Save config
            save_config(config, cfg_path, schemes, wbits, abits)

            result = {
                "schemes": schemes,
                "weight_bits": wbits,
                "act_bits": abits,
                "time": elapsed,
                "peak_gb": peak_gb,
                "scheme_distribution": scheme_counts,
                "status": "OK",
            }
            results.append(result)
            print(
                f"\n  W{wbits}avg A{abits}avg: {elapsed:.1f}s, "
                f"{peak_gb:.2f}GB, {scheme_counts}"
            )

    # Summary
    print(f"\n{'='*80}")
    print(f"SWEEP SUMMARY: {model_short} | Schemes: {schemes}")
    print(f"{'='*80}")
    print(
        f"{'W-bits':>8} {'A-bits':>8} {'Time(s)':>8} {'Peak(GB)':>9} "
        f"{'Status':>10} {'Distribution'}"
    )
    print("-" * 80)
    for r in results:
        abits_str = str(r["act_bits"]) if r["act_bits"] is not None else "none"
        print(
            f"{r['weight_bits']:>8.2f} {abits_str:>8} "
            f"{r.get('time', 0):>8.1f} {r.get('peak_gb', 0):>9.2f} "
            f"{r['status']:>10} {r.get('scheme_distribution', '')}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WNaM activation quantization sweep for HIGGS"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--short", required=True, help="Short name for save directory")
    parser.add_argument(
        "--schemes",
        nargs="+",
        required=True,
        help="Candidate quantization schemes (e.g., W4A16 W8A16 W4A8 W8A8)",
    )
    parser.add_argument(
        "--weight-bits",
        type=float,
        nargs="+",
        default=[4.5, 5.0, 5.5, 6.0],
        help="Target average weight bitwidths",
    )
    parser.add_argument(
        "--act-bits",
        type=float,
        nargs="+",
        default=[None],
        help="Target average activation bitwidths (omit for no constraint)",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Generate configs only, skip quantization (Phase 1 only)",
    )
    args = parser.parse_args()

    # Parse act-bits: allow "none" as string
    act_bits = []
    for ab in args.act_bits:
        if ab is None or (isinstance(ab, str) and ab.lower() == "none"):
            act_bits.append(None)
        else:
            act_bits.append(float(ab))

    run_sweep(
        args.model,
        args.short,
        args.schemes,
        args.weight_bits,
        act_bits,
        config_only=args.config_only,
    )
