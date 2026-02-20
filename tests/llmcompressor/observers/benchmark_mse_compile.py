"""
E2E benchmark: MSE observer torch.compile optimization.

Quantizes a model with FP8 W8A8 using the MSE observer via oneshot(),
comparing the baseline path (enable_torch_compile=False) against the
compiled path (enable_torch_compile=True). Measures wall-clock time,
peak memory, and numerical equivalence of weight scales.

Usage:
    python tests/llmcompressor/observers/benchmark_mse_compile.py
    python tests/llmcompressor/observers/benchmark_mse_compile.py \
        --num-calibration-samples 128 --output-file results.md
"""

import argparse
import gc
import sys
import time
import tracemalloc
from collections import OrderedDict

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot, reset_session
from llmcompressor.modifiers.quantization import QuantizationModifier

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_DATASET = "open_platypus"
DEFAULT_NUM_SAMPLES = 64
DEFAULT_MAX_SEQ_LENGTH = 384


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark MSE observer torch.compile optimization"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="Model to quantize"
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET, help="Calibration dataset"
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save markdown results",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for scale comparison",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for scale comparison",
    )
    return parser.parse_args()


def build_recipe(enable_torch_compile: bool) -> QuantizationModifier:
    """Build FP8 W8A8 recipe with MSE observer and the given compile flag.

    Mirrors the FP8_DYNAMIC preset but uses explicit config_groups so that
    observer_kwargs can be passed through to the weight observer.
    """
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type="float",
            symmetric=True,
            strategy="channel",
            dynamic=False,
            observer="memoryless_mse",
            observer_kwargs={"enable_torch_compile": enable_torch_compile},
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type="float",
            symmetric=True,
            strategy="token",
            dynamic=True,
        ),
    )
    return QuantizationModifier(
        targets=["Linear"],
        ignore=["lm_head"],
        config_groups={"group_0": scheme},
    )


def extract_weight_scales(model) -> OrderedDict:
    """Collect weight_scale tensors from all quantized modules."""
    scales = OrderedDict()
    for name, module in model.named_modules():
        weight_scale = getattr(module, "weight_scale", None)
        if weight_scale is not None:
            scales[name] = weight_scale.detach().clone().cpu().float()
    return scales


def run_oneshot(args, enable_torch_compile: bool) -> dict:
    """Run oneshot quantization and measure time + memory.

    Returns dict with elapsed_s, peak_memory_mb, and weight_scales.
    """
    label = "compiled" if enable_torch_compile else "baseline"
    use_cuda = torch.cuda.is_available()

    # Clean slate
    reset_session()
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"\n{'='*60}")
    print(f"  Running {label} (enable_torch_compile={enable_torch_compile})")
    print(f"{'='*60}")

    # Load fresh model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    recipe = build_recipe(enable_torch_compile)

    # Start memory tracking
    if not use_cuda:
        tracemalloc.start()

    # Time the oneshot call
    start = time.perf_counter()
    model = oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset=args.dataset,
        num_calibration_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
    )
    elapsed = time.perf_counter() - start

    # Measure peak memory
    if use_cuda:
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mb = peak_bytes / (1024 * 1024)

    # Extract weight scales before cleanup
    weight_scales = extract_weight_scales(model)

    print(f"  Time: {elapsed:.1f}s | Peak memory: {peak_memory_mb:.0f} MB")
    print(f"  Quantized layers: {len(weight_scales)}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    return {
        "elapsed_s": elapsed,
        "peak_memory_mb": peak_memory_mb,
        "weight_scales": weight_scales,
    }


def compare_scales(
    baseline_scales: OrderedDict,
    compiled_scales: OrderedDict,
    atol: float,
    rtol: float,
) -> dict:
    """Compare weight scales between baseline and compiled runs."""
    assert set(baseline_scales.keys()) == set(
        compiled_scales.keys()
    ), "Mismatch in quantized layer names between baseline and compiled"

    num_layers = len(baseline_scales)
    num_matching = 0
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    mismatched_layers = []

    for name in baseline_scales:
        s_base = baseline_scales[name]
        s_comp = compiled_scales[name]

        abs_diff = (s_base - s_comp).abs().max().item()
        denom = s_base.abs().max().item()
        rel_diff = abs_diff / (denom + 1e-12) if denom > 0 else 0.0

        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)

        if torch.allclose(s_base, s_comp, atol=atol, rtol=rtol):
            num_matching += 1
        else:
            mismatched_layers.append(name)

    return {
        "all_close": num_matching == num_layers,
        "num_layers": num_layers,
        "num_matching": num_matching,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mismatched_layers": mismatched_layers,
    }


def format_results(args, baseline: dict, compiled: dict, equiv: dict) -> str:
    """Format benchmark results as a markdown table."""
    b_time = baseline["elapsed_s"]
    c_time = compiled["elapsed_s"]
    b_mem = baseline["peak_memory_mb"]
    c_mem = compiled["peak_memory_mb"]

    time_delta = (c_time - b_time) / b_time * 100
    mem_delta = (c_mem - b_mem) / b_mem * 100 if b_mem > 0 else 0.0
    speedup = b_time / c_time if c_time > 0 else float("inf")

    lines = [
        "## MSE Observer torch.compile Benchmark Results",
        "",
        "| Metric | Baseline | Compiled | Delta |",
        "|---|---|---|---|",
        f"| Time (s) | {b_time:.1f} | {c_time:.1f} | {time_delta:+.1f}% |",
        f"| Peak memory (MB) | {b_mem:.0f} | {c_mem:.0f} | {mem_delta:+.1f}% |",
        f"| Quantized layers | {equiv['num_layers']} | {equiv['num_layers']} | - |",
        f"| Scales all_close | - | - | {equiv['all_close']} |",
        f"| Max abs diff | - | - | {equiv['max_abs_diff']:.2e} |",
        f"| Max rel diff | - | - | {equiv['max_rel_diff']:.2e} |",
        f"| Matching layers | - | - | {equiv['num_matching']}/{equiv['num_layers']} |",
        "",
        f"**Speedup**: {speedup:.2f}x",
        "",
        f"**Config**: model={args.model}, dataset={args.dataset}, "
        f"samples={args.num_calibration_samples}, "
        f"max_seq_length={args.max_seq_length}, "
        f"atol={args.atol}, rtol={args.rtol}",
    ]

    if equiv["mismatched_layers"]:
        lines.append("")
        lines.append(f"**Mismatched layers** ({len(equiv['mismatched_layers'])}):")
        for name in equiv["mismatched_layers"][:10]:
            lines.append(f"- `{name}`")
        if len(equiv["mismatched_layers"]) > 10:
            lines.append(f"- ... and {len(equiv['mismatched_layers']) - 10} more")

    return "\n".join(lines)


def main():
    args = parse_args()

    print("MSE Observer torch.compile Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Calibration samples: {args.num_calibration_samples}")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    baseline = run_oneshot(args, enable_torch_compile=False)
    compiled = run_oneshot(args, enable_torch_compile=True)

    equiv = compare_scales(
        baseline["weight_scales"],
        compiled["weight_scales"],
        atol=args.atol,
        rtol=args.rtol,
    )

    results = format_results(args, baseline, compiled, equiv)

    print(f"\n{'='*60}")
    print(results)
    print(f"{'='*60}")

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(results + "\n")
        print(f"\nResults saved to {args.output_file}")

    sys.exit(0 if equiv["all_close"] else 1)


if __name__ == "__main__":
    main()
