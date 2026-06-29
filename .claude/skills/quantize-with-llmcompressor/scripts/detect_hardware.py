#!/usr/bin/env python3
"""Detect local accelerators and report which quantization schemes they can
*efficiently serve* and *calibrate*, for use with LLM Compressor.

The report distinguishes two things that are commonly conflated:

* What your GPU can **calibrate / produce** -- almost any GPU (or even CPU with
  offloading) can run ``oneshot`` to emit a checkpoint in any format.
* What your GPU can **efficiently serve** in vLLM -- this is gated by the
  hardware datapath (FP4 needs Blackwell, FP8 needs Ada/Hopper+, INT8 needs
  Turing+). Quantizing for a *different* deployment GPU is fine and common.

Run directly for a human-readable report, or ``--json`` for machine output.
"""

import argparse
import json
import shutil
import subprocess
import sys

# Compute-capability major.minor -> (architecture, native datapaths).
# "native" means a dedicated hardware datapath exists for that numeric type;
# weight-only schemes (W4A16/W8A16) run on essentially any CUDA GPU via Marlin
# kernels regardless of this table.
_ARCH_TABLE = [
    # (cc_major, cc_minor_min, cc_minor_max, arch, native_types)
    (12, 0, 99, "Blackwell", {"fp4", "fp8", "int8"}),
    (10, 0, 99, "Blackwell", {"fp4", "fp8", "int8"}),
    (9, 0, 99, "Hopper", {"fp8", "int8"}),
    (8, 9, 9, "Ada Lovelace", {"fp8", "int8"}),
    (8, 0, 6, "Ampere", {"int8"}),
    (7, 5, 5, "Turing", {"int8"}),
    (7, 0, 2, "Volta", {"int8"}),
]


# Curated specs for common *deployment* targets, so the skill can recommend a
# scheme for a GPU you are not physically on (e.g. quantizing on a laptop to
# deploy on an H100). Only architecture, VRAM and native datapaths are listed --
# those are well-established and are all the scheme recommendation needs. Browse
# the community hardware catalog at https://huggingface.co/hardware for more.
# key -> (display_name, architecture, vram_gb, native_datapaths)
TARGET_GPUS = {
    "b200": ("NVIDIA B200", "Blackwell", 192, {"fp4", "fp8", "int8"}),
    "gb200": ("NVIDIA GB200", "Blackwell", 192, {"fp4", "fp8", "int8"}),
    "rtx5090": ("NVIDIA RTX 5090", "Blackwell", 32, {"fp4", "fp8", "int8"}),
    "rtx5090-laptop": (
        "NVIDIA RTX 5090 Laptop",
        "Blackwell",
        24,
        {"fp4", "fp8", "int8"},
    ),
    "h200": ("NVIDIA H200", "Hopper", 141, {"fp8", "int8"}),
    "h100": ("NVIDIA H100", "Hopper", 80, {"fp8", "int8"}),
    "l40s": ("NVIDIA L40S", "Ada Lovelace", 48, {"fp8", "int8"}),
    "l4": ("NVIDIA L4", "Ada Lovelace", 24, {"fp8", "int8"}),
    "rtx4090": ("NVIDIA RTX 4090", "Ada Lovelace", 24, {"fp8", "int8"}),
    "a100": ("NVIDIA A100 80GB", "Ampere", 80, {"int8"}),
    "a100-40": ("NVIDIA A100 40GB", "Ampere", 40, {"int8"}),
    "a10": ("NVIDIA A10", "Ampere", 24, {"int8"}),
    "rtx3090": ("NVIDIA RTX 3090", "Ampere", 24, {"int8"}),
    "v100": ("NVIDIA V100", "Volta", 32, {"int8"}),
    "t4": ("NVIDIA T4", "Turing", 16, {"int8"}),
}


def _normalize_target(name):
    return name.lower().replace(" ", "").replace("_", "").replace("nvidia", "")


def lookup_target(name):
    """Return a GPU record for a named deployment target, or None."""
    key = _normalize_target(name)
    aliases = {
        "5090": "rtx5090",
        "4090": "rtx4090",
        "3090": "rtx3090",
        "a100-80": "a100",
        "gb200nvl": "gb200",
    }
    key = aliases.get(key, key)
    if key not in TARGET_GPUS:
        return None
    display, arch, vram_gb, native = TARGET_GPUS[key]
    return {
        "index": None,
        "name": display,
        "compute_capability": "n/a (target spec)",
        "architecture": arch,
        "vram_gb": vram_gb,
        "native_datapaths": sorted(native),
        "serve_schemes": _schemes_for(native),
        "recommended": _recommend(arch, native, vram_gb),
        "max_model_guidance": _vram_guidance(vram_gb),
    }


def _classify(cc_major, cc_minor):
    for major, lo, hi, arch, native in _ARCH_TABLE:
        if cc_major == major and lo <= cc_minor <= hi:
            return arch, native
    if cc_major >= 10:  # future arch, assume superset of Blackwell
        return "Blackwell+", {"fp4", "fp8", "int8"}
    return "Pre-Volta", set()


def _schemes_for(native):
    """Map native datapaths to LLM Compressor schemes you can *efficiently
    serve* on this GPU, with a one-line rationale each."""
    serve = []
    # Weight-only always works (Marlin kernels), runs anywhere.
    serve.append(("W4A16", "weight-only INT4, runs on any GPU; max compression"))
    serve.append(("W8A16", "weight-only INT8, runs on any GPU; safest accuracy"))
    if "int8" in native:
        serve.append(("W8A8 (INT8)", "native INT8 tensor cores; high-QPS serving"))
    if "fp8" in native:
        serve.append(("FP8_DYNAMIC", "native FP8; best perf/accuracy balance"))
        serve.append(("FP8_BLOCK", "native FP8 block-wise; large/MoE models"))
    if "fp4" in native:
        serve.append(("NVFP4", "native FP4 (W4A4); max throughput on Blackwell"))
        serve.append(("NVFP4A16", "FP4 weights, FP16 acts; data-free, accurate"))
        serve.append(("MXFP4", "OCP microscale FP4 weight-only"))
    return serve


def _recommend(arch, native, vram_gb):
    """Single best default scheme for this GPU as a serving target."""
    if "fp4" in native:
        return "NVFP4", (
            "Blackwell has native FP4 tensor cores -- NVFP4 gives the highest "
            "throughput and ~3.5x compression with strong accuracy."
        )
    if "fp8" in native:
        return "FP8_DYNAMIC", (
            "Ada/Hopper have native FP8 -- FP8_DYNAMIC is data-free (RTN), ~2x "
            "smaller, and the best accuracy/perf balance. Use W4A16 if you need "
            "more compression for a memory-bound / low-QPS workload."
        )
    if "int8" in native:
        return "W4A16", (
            "No native FP8/FP4 here. W4A16 (GPTQ/AWQ) gives ~3.7x compression "
            "and runs via Marlin kernels; INT8 W8A8 is the alternative for "
            "high-QPS throughput."
        )
    return "W4A16", (
        "Older/CPU target -- weight-only W4A16 is the broadly supported option."
    )


def _vram_guidance(vram_gb):
    """Rough largest model you can quantize with naive oneshot on one GPU.
    Sequential onloading (see big_models_with_sequential_onloading) lets you go
    far larger by streaming layers."""
    if vram_gb <= 0:
        return "unknown"
    # oneshot holds the model (~2 bytes/param bf16) plus activations/overhead.
    naive_b = vram_gb / 2.6
    return (
        f"~{naive_b:.0f}B params via naive oneshot; much larger with "
        f"sequential onloading or model_free_ptq (FP8 data-free)."
    )


def detect():
    info = {
        "gpus": [],
        "cuda_available": False,
        "torch_version": None,
        "notes": [],
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                cc_major, cc_minor = torch.cuda.get_device_capability(i)
                arch, native = _classify(cc_major, cc_minor)
                vram_gb = round(props.total_memory / (1024**3), 1)
                info["gpus"].append(
                    {
                        "index": i,
                        "name": props.name,
                        "compute_capability": f"{cc_major}.{cc_minor}",
                        "architecture": arch,
                        "vram_gb": vram_gb,
                        "native_datapaths": sorted(native),
                        "serve_schemes": _schemes_for(native),
                        "recommended": _recommend(arch, native, vram_gb),
                        "max_model_guidance": _vram_guidance(vram_gb),
                    }
                )
    except ImportError:
        info["notes"].append(
            "PyTorch not importable. Falling back to nvidia-smi for raw specs; "
            "install torch (cu128 build for Blackwell) to enable quantization."
        )
        _nvidia_smi_fallback(info)
    return info


def _nvidia_smi_fallback(info):
    if not shutil.which("nvidia-smi"):
        info["notes"].append("nvidia-smi not found; no NVIDIA GPU detected.")
        return
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        info["notes"].append(f"nvidia-smi failed: {exc}")
        return
    for i, line in enumerate(out.strip().splitlines()):
        name, mem, cc = (p.strip() for p in line.split(","))
        cc_major, cc_minor = (int(x) for x in cc.split("."))
        arch, native = _classify(cc_major, cc_minor)
        vram_gb = round(float(mem) / 1024, 1)
        info["gpus"].append(
            {
                "index": i,
                "name": name,
                "compute_capability": cc,
                "architecture": arch,
                "vram_gb": vram_gb,
                "native_datapaths": sorted(native),
                "serve_schemes": _schemes_for(native),
                "recommended": _recommend(arch, native, vram_gb),
                "max_model_guidance": _vram_guidance(vram_gb),
            }
        )


def _print_report(info):
    print("=" * 70)
    print("LLM Compressor -- Hardware Capability Report")
    print("=" * 70)
    if info["torch_version"]:
        print(
            f"torch: {info['torch_version']}  cuda_available: {info['cuda_available']}"
        )
    if not info["gpus"]:
        print("\nNo CUDA GPU detected.")
        print("You can still quantize on CPU (slow) and produce checkpoints for")
        print("any deployment target; weight-only W4A16 is the safe default.")
        for note in info["notes"]:
            print(f"  - {note}")
        return
    for g in info["gpus"]:
        label = f"GPU {g['index']}" if g["index"] is not None else "Target"
        print(f"\n{label}: {g['name']}")
        print(
            f"  Architecture     : {g['architecture']} "
            f"(compute {g['compute_capability']})"
        )
        print(f"  VRAM             : {g['vram_gb']} GB")
        print(
            f"  Native datapaths : {', '.join(g['native_datapaths']) or 'weight-only'}"
        )
        rec_scheme, rec_why = g["recommended"]
        print(f"  >> Recommended   : {rec_scheme}")
        print(f"     {rec_why}")
        print(f"  Model size guide : {g['max_model_guidance']}")
        print("  Efficiently servable schemes on THIS GPU:")
        for scheme, why in g["serve_schemes"]:
            print(f"     - {scheme:<14} {why}")
    for note in info["notes"]:
        print(f"\nNote: {note}")
    print("\nReminder: you can quantize FOR a different deployment GPU than the")
    print("one you calibrate on. Pick the scheme for your *serving* hardware.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable")
    parser.add_argument(
        "--target",
        help="report specs for a named deployment GPU instead of the local one "
        "(e.g. h100, b200, a100, rtx4090). Use when quantizing off-machine.",
    )
    parser.add_argument(
        "--list-targets", action="store_true", help="list known target GPUs"
    )
    args = parser.parse_args()

    if args.list_targets:
        print("Known deployment targets (--target <name>):")
        for key, (display, arch, vram, _native) in sorted(TARGET_GPUS.items()):
            print(f"  {key:<16} {display} ({arch}, {vram} GB)")
        return

    if args.target:
        rec = lookup_target(args.target)
        if rec is None:
            print(
                f"Unknown target '{args.target}'. Run --list-targets to see "
                f"known GPUs, or read specs from https://huggingface.co/hardware"
            )
            sys.exit(1)
        info = {
            "gpus": [rec],
            "cuda_available": False,
            "torch_version": None,
            "notes": [f"Target spec for {rec['name']} (not the local GPU)."],
        }
        if args.json:
            json.dump(info, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            _print_report(info)
        return

    info = detect()
    if args.json:
        json.dump(info, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_report(info)


if __name__ == "__main__":
    main()
