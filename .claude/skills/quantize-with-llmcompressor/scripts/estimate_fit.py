#!/usr/bin/env python3
"""Estimate whether a model fits a GPU once quantized -- the "will it run?"
calculation that pairs with hardware detection.

Given a model's real parameter count (read from Hugging Face safetensors
metadata, no download) and a target scheme, it estimates:

* the quantized weight footprint on disk / in VRAM,
* the VRAM needed to *serve* it (weights + headroom for KV cache + activations),
* the VRAM needed to *quantize* it with naive ``oneshot`` vs sequential
  onloading,

then checks each against the available VRAM -- the local GPU (auto-detected),
a named deployment target (``--target h100``), or an explicit ``--vram-gb``.

This mirrors the spirit of Hugging Face's per-hardware model-fit panel
(https://huggingface.co/docs/hub/main/hardware) but for LLM Compressor's
compressed-tensors / vLLM schemes rather than GGUF/MLX.

Examples:
    python estimate_fit.py --model meta-llama/Llama-3.1-8B-Instruct --scheme NVFP4
    python estimate_fit.py --model Qwen/Qwen2.5-72B-Instruct --scheme W4A16 \
        --target a100
    python estimate_fit.py --params 8030261248 --scheme FP8_DYNAMIC --vram-gb 24
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_hardware import detect, lookup_target  # noqa: E402

# Effective bytes per weight parameter, including group-scale overhead. Only
# Linear weights are quantized (embeddings / lm_head / norms stay 16-bit), so
# treat these as a ~+/-10% estimate on the dominant weight term.
SCHEME_BYTES = {
    "BF16": 2.0,
    "FP16": 2.0,
    "FP8_DYNAMIC": 1.0,
    "FP8": 1.0,
    "FP8_BLOCK": 1.0,
    "W8A8": 1.0,
    "INT8": 1.0,
    "W8A16": 1.0,
    "W4A16": 0.52,
    "W8A16_ASYM": 1.0,
    "NVFP4": 0.56,
    "NVFP4A16": 0.56,
    "MXFP4": 0.55,
    "MXFP4A16": 0.55,
}

GB = 1024**3


def param_count(model_id):
    """Exact parameter count from HF safetensors metadata (no download)."""
    from huggingface_hub import model_info

    try:
        info = model_info(model_id)
    except Exception as exc:  # noqa: BLE001 - surface a friendly, actionable msg
        raise ValueError(
            f"Failed to fetch model info for '{model_id}' from the HF Hub: "
            f"{exc}. If this is a gated model (e.g. Llama), log in "
            "(`huggingface-cli login` or set HF_TOKEN), or pass the parameter "
            "count explicitly with --params N."
        ) from exc
    st = getattr(info, "safetensors", None)
    if st and st.total:
        return int(st.total)
    raise ValueError(
        f"Could not read safetensors param count for '{model_id}'. "
        f"Pass --params N explicitly."
    )


def available_vram_gb(args):
    if args.vram_gb is not None:
        return args.vram_gb, f"explicit --vram-gb {args.vram_gb}"
    if args.target:
        rec = lookup_target(args.target)
        if rec is None:
            raise SystemExit(
                f"Unknown target '{args.target}'. "
                f"Run detect_hardware.py --list-targets."
            )
        return rec["vram_gb"], f"target {rec['name']}"
    info = detect()
    if info["gpus"]:
        g = info["gpus"][0]
        return g["vram_gb"], f"local {g['name']}"
    raise SystemExit("No local GPU detected; pass --target or --vram-gb.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="HF model id (param count read from the Hub)")
    g.add_argument("--params", type=int, help="explicit parameter count")
    parser.add_argument("--scheme", required=True, help="target quant scheme")
    parser.add_argument(
        "--target", help="named deployment GPU (see detect_hardware.py --list-targets)"
    )
    parser.add_argument("--vram-gb", type=float, help="explicit available VRAM")
    args = parser.parse_args()

    scheme = args.scheme.upper()
    bpp = SCHEME_BYTES.get(scheme)
    if bpp is None:
        raise SystemExit(
            f"Unknown scheme '{scheme}'. Known: {', '.join(sorted(SCHEME_BYTES))}"
        )

    n = args.params if args.params else param_count(args.model)
    vram_gb, vram_src = available_vram_gb(args)

    quant_gb = n * bpp / GB
    baseline_gb = n * 2.0 / GB
    serve_gb = quant_gb * 1.3  # weights + KV cache / activation headroom
    calib_naive_gb = baseline_gb * 1.15  # BF16 load + calibration overhead
    # Sequential onloading keeps ~one transformer block resident at a time.
    calib_seq_gb = max(baseline_gb / 20.0, 2.0)

    name = args.model or f"{n / 1e9:.1f}B params"
    print("=" * 66)
    print(f"Fit estimate: {name}  scheme={scheme}")
    print(f"Available VRAM: {vram_gb} GB  ({vram_src})")
    print("=" * 66)
    print(f"  parameters           : {n / 1e9:.2f} B")
    print(f"  BF16 baseline weights: {baseline_gb:.1f} GB")
    print(
        f"  quantized weights    : {quant_gb:.1f} GB  "
        f"({baseline_gb / quant_gb:.1f}x smaller)"
    )
    print(f"  est. serving VRAM    : {serve_gb:.1f} GB  (weights + KV/act headroom)")
    print(
        f"  est. calibrate VRAM  : {calib_naive_gb:.1f} GB naive oneshot | "
        f"~{calib_seq_gb:.1f} GB sequential onloading"
    )
    print("-" * 66)

    def verdict(need, what):
        if need <= vram_gb:
            print(f"  [OK]   {what}: needs ~{need:.1f} GB, fits in {vram_gb} GB")
            return True
        print(f"  [TIGHT/NO] {what}: needs ~{need:.1f} GB > {vram_gb} GB")
        return False

    serve_ok = verdict(serve_gb, "serve quantized")
    if not verdict(calib_naive_gb, "quantize (naive oneshot)"):
        if calib_seq_gb <= vram_gb:
            print(
                "         -> use sequential onloading or model_free_ptq "
                "(data-free schemes) to quantize within this VRAM."
            )
        else:
            print(
                "         -> too large for one GPU; use multi-GPU / disk "
                "offload (see big_models_with_sequential_onloading)."
            )
    print("-" * 66)
    if not serve_ok:
        print(
            "  Tip: a lower-bit scheme reduces serving VRAM. Try W4A16/NVFP4 "
            "if you are above FP8, or split across GPUs in vLLM (-tp N)."
        )
    else:
        print(f"  {name} in {scheme} should serve comfortably on {vram_gb} GB.")
    print(
        "  Note: estimates are +/-10%; KV-cache size grows with context and "
        "batch. Validate with a short vLLM run."
    )


if __name__ == "__main__":
    main()
