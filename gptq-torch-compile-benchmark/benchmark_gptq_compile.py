#!/usr/bin/env python3
"""
GPTQ torch.compile Benchmark Script

Compares baseline vs compiled performance using the built-in enable_torch_compile flag.

Metrics collected:
- Total quantization wall time (cold and warm)
- Compile time overhead
- Peak GPU memory
- Numerical correctness (model output comparison)
- Storage-size proof (logical vs storage size for views)

Usage:
    python benchmark_gptq_compile.py --scenario tinyllama --mode baseline
    python benchmark_gptq_compile.py --scenario tinyllama --mode compiled
    python benchmark_gptq_compile.py --scenario qwen3b --mode both
    python benchmark_gptq_compile.py --scenario tinyllama --mode numerical_check
    python benchmark_gptq_compile.py --scenario tinyllama --mode memory_sentinel
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

# Add local src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from benchmark_utils import (
    Timer,
    format_time,
    generate_markdown_table,
    generate_stress_test_conclusion,
    get_environment_info,
    get_gpu_memory_stats,
    get_compile_stats,
    reset_dynamo_counters,
    compute_variance_stats,
    print_header,
    reset_gpu_state,
    save_results,
    compare_model_weights,
    compare_model_outputs,
    capture_storage_size_proof,
    run_memory_sentinel,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier


# Scenario configurations
SCENARIOS = {
    "tinyllama": {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "num_calibration_samples": 256,
        "max_seq_length": 512,
        "description": "TinyLlama 1.1B - fast iteration",
    },
    "qwen3b": {
        "model": "Qwen/Qwen2.5-3B",
        "num_calibration_samples": 256,
        "max_seq_length": 512,
        "description": "Qwen2.5-3B - production representative",
    },
    "blocksize_64": {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "num_calibration_samples": 256,
        "max_seq_length": 512,
        "block_size": 64,
        "description": "TinyLlama with blocksize=64 (shape stress test)",
    },
    "blocksize_256": {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "num_calibration_samples": 256,
        "max_seq_length": 512,
        "block_size": 256,
        "description": "TinyLlama with blocksize=256 (shape stress test)",
    },
}


def run_quantization(
    scenario_config: dict,
    enable_compile: bool,
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict:
    """
    Run GPTQ quantization and collect metrics.

    Args:
        scenario_config: Dict with model, samples, etc.
        enable_compile: Whether to use torch.compile
        output_dir: Where to save model (None = don't save)
        seed: Random seed for reproducibility

    Returns:
        Dict with timing and memory metrics
    """
    reset_gpu_state()

    # Set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build recipe
    recipe_kwargs = {
        "targets": "Linear",
        "scheme": "W4A16",
        "ignore": ["lm_head"],
        "enable_torch_compile": enable_compile,
    }
    if "block_size" in scenario_config:
        recipe_kwargs["block_size"] = scenario_config["block_size"]

    recipe = GPTQModifier(**recipe_kwargs)

    # Time the quantization
    with Timer() as timer:
        try:
            oneshot(
                model=scenario_config["model"],
                dataset="open_platypus",
                recipe=recipe,
                output_dir=str(output_dir) if output_dir else None,
                max_seq_length=scenario_config["max_seq_length"],
                num_calibration_samples=scenario_config["num_calibration_samples"],
            )
            status = "success"
            error = None
        except Exception as e:
            status = "error"
            error = str(e)
            import traceback
            traceback.print_exc()

    # Collect metrics
    memory = get_gpu_memory_stats()

    return {
        "status": status,
        "error": error,
        "wall_time_seconds": timer.elapsed,
        "peak_memory": memory,
        "enable_torch_compile": enable_compile,
    }


def run_baseline(scenario: str, output_dir: Path, num_runs: int = 3) -> dict:
    """Run baseline (no compile) benchmark."""
    print_header(f"BASELINE BENCHMARK: {scenario}")

    config = SCENARIOS[scenario]
    print(f"Model: {config['model']}")
    print(f"Calibration samples: {config['num_calibration_samples']}")
    print(f"Runs: {num_runs}")

    results = {
        "scenario": scenario,
        "config": config,
        "mode": "baseline",
        "times": [],
        "peak_memory_runs": [],  # Per-run memory stats
        "peak_memory": None,  # Max across runs
    }

    # Warmup run (not measured)
    print("\n[Warmup run - not measured]")
    _ = run_quantization(config, enable_compile=False)

    # Measured runs
    for i in range(num_runs):
        print(f"\n[Run {i+1}/{num_runs}]")
        run_result = run_quantization(config, enable_compile=False)

        if run_result["status"] != "success":
            results["error"] = run_result["error"]
            break

        results["times"].append(run_result["wall_time_seconds"])
        results["peak_memory_runs"].append(run_result["peak_memory"])
        print(f"  Time: {format_time(run_result['wall_time_seconds'])}")
        print(f"  Peak GPU: {run_result['peak_memory']['max_allocated_gb']:.2f} GB")

    if results["times"]:
        results["mean_time"] = sum(results["times"]) / len(results["times"])
        results["variance_stats"] = compute_variance_stats(results["times"])
        # Compute max memory across all runs
        max_alloc = max(r["max_allocated_gb"] for r in results["peak_memory_runs"])
        max_reserved = max(r["max_reserved_gb"] for r in results["peak_memory_runs"])
        results["peak_memory"] = {
            "max_allocated_gb": max_alloc,
            "max_reserved_gb": max_reserved,
        }
        results["status"] = "success"
    else:
        results["status"] = "error"

    save_results(results, output_dir / "baseline.json")
    return results


def run_compiled(scenario: str, output_dir: Path, num_warm_runs: int = 3) -> dict:
    """Run compiled benchmark with cold and warm measurements."""
    print_header(f"COMPILED BENCHMARK: {scenario}")

    config = SCENARIOS[scenario]
    print(f"Model: {config['model']}")
    print(f"Calibration samples: {config['num_calibration_samples']}")
    print(f"Warm runs: {num_warm_runs}")

    results = {
        "scenario": scenario,
        "config": config,
        "mode": "compiled",
        "cold_time": None,
        "warm_times": [],
        "peak_memory_runs": [],  # Per-run memory stats
        "peak_memory": None,  # Max across runs
        "compile_stats": None,
    }

    # Reset dynamo counters before cold run
    reset_dynamo_counters()
    torch._dynamo.reset()

    # Cold run (includes compilation)
    print("\n[Cold run - includes compilation]")
    cold_result = run_quantization(config, enable_compile=True)

    # Capture compile stats after cold run
    results["compile_stats"] = get_compile_stats()
    print(f"  Compile stats: {results['compile_stats'].get('num_graphs', 0)} graphs, "
          f"{results['compile_stats'].get('graph_breaks', 0)} graph breaks")

    if cold_result["status"] != "success":
        results["status"] = "error"
        results["error"] = cold_result["error"]
        save_results(results, output_dir / "compiled.json")
        return results

    results["cold_time"] = cold_result["wall_time_seconds"]
    results["cold_peak_memory"] = cold_result["peak_memory"]
    results["peak_memory_runs"].append({"run": "cold", **cold_result["peak_memory"]})
    print(f"  Time: {format_time(cold_result['wall_time_seconds'])}")
    print(f"  Peak GPU: {cold_result['peak_memory']['max_allocated_gb']:.2f} GB")

    # Warm runs (compile cache should be warm)
    for i in range(num_warm_runs):
        print(f"\n[Warm run {i+1}/{num_warm_runs}]")
        warm_result = run_quantization(config, enable_compile=True)

        if warm_result["status"] != "success":
            results["error"] = warm_result["error"]
            break

        results["warm_times"].append(warm_result["wall_time_seconds"])
        results["peak_memory_runs"].append({"run": f"warm_{i+1}", **warm_result["peak_memory"]})
        print(f"  Time: {format_time(warm_result['wall_time_seconds'])}")
        print(f"  Peak GPU: {warm_result['peak_memory']['max_allocated_gb']:.2f} GB")

    # Calculate summary stats
    if results["warm_times"]:
        results["warm_mean"] = sum(results["warm_times"]) / len(results["warm_times"])
        results["compile_overhead"] = results["cold_time"] - results["warm_mean"]
        results["variance_stats"] = compute_variance_stats(results["warm_times"])
        # Compute max memory across all runs (cold + warm)
        max_alloc = max(r["max_allocated_gb"] for r in results["peak_memory_runs"])
        max_reserved = max(r["max_reserved_gb"] for r in results["peak_memory_runs"])
        results["peak_memory"] = {
            "max_allocated_gb": max_alloc,
            "max_reserved_gb": max_reserved,
        }
        results["status"] = "success"

        # Print variance stats
        var_stats = results["variance_stats"]
        print(f"\n[Variance Stats]")
        print(f"  Mean: {format_time(var_stats['mean'])}")
        print(f"  Stddev: {var_stats['stddev']:.1f}s")
        print(f"  CV: {var_stats['cv']:.1f}%")
    else:
        results["status"] = "partial"

    save_results(results, output_dir / "compiled.json")
    return results


def capture_storage_proof_from_path(model_path: Path) -> dict:
    """Load a saved model and capture storage-size proof."""
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",  # CPU to avoid GPU memory issues
            torch_dtype=torch.float16,
        )
        proof = capture_storage_size_proof(model)
        del model
        return proof
    except Exception as e:
        return {"error": str(e)}


def run_numerical_check(scenario: str, output_dir: Path) -> dict:
    """
    Compare model outputs between baseline and compiled runs.

    Uses output comparison (logits and generated tokens) rather than raw weight
    comparison, because packed int32 representations give meaningless diffs.

    Also captures storage-size proof for both models.
    """
    print_header(f"NUMERICAL CHECK: {scenario}")

    config = SCENARIOS[scenario]
    baseline_path = output_dir / "weights_baseline"
    compiled_path = output_dir / "weights_compiled"

    # Clean up old results
    shutil.rmtree(baseline_path, ignore_errors=True)
    shutil.rmtree(compiled_path, ignore_errors=True)

    # Run baseline and save weights
    print("\n[1/4] Running baseline quantization...")
    _ = run_quantization(config, enable_compile=False, output_dir=baseline_path)

    # Capture storage-size proof for baseline
    print("\n[2/4] Capturing storage-size proof for baseline...")
    baseline_storage_proof = capture_storage_proof_from_path(baseline_path)
    print(f"  Total logical: {baseline_storage_proof.get('total_logical_mb', 'N/A')} MB")
    print(f"  Total storage: {baseline_storage_proof.get('total_storage_mb', 'N/A')} MB")
    print(f"  Has views: {baseline_storage_proof.get('has_views', 'N/A')}")

    # Reset for compiled run
    reset_gpu_state()
    torch._dynamo.reset()  # Reset compile cache for fair comparison

    # Run compiled and save weights
    print("\n[3/4] Running compiled quantization...")
    _ = run_quantization(config, enable_compile=True, output_dir=compiled_path)

    # Capture storage-size proof for compiled
    print("Capturing storage-size proof for compiled...")
    compiled_storage_proof = capture_storage_proof_from_path(compiled_path)
    print(f"  Total logical: {compiled_storage_proof.get('total_logical_mb', 'N/A')} MB")
    print(f"  Total storage: {compiled_storage_proof.get('total_storage_mb', 'N/A')} MB")
    print(f"  Has views: {compiled_storage_proof.get('has_views', 'N/A')}")

    # Compare model outputs (not raw weights)
    print("\n[4/4] Comparing model outputs...")
    comparison = compare_model_outputs(
        baseline_path,
        compiled_path,
        num_samples=5,
        max_new_tokens=20,
    )

    results = {
        "scenario": scenario,
        "mode": "numerical_check",
        "storage_size_proof": {
            "baseline": baseline_storage_proof,
            "compiled": compiled_storage_proof,
        },
        **comparison,
    }

    print(f"\nResults (Output Comparison):")
    if comparison.get("status") == "success":
        print(f"  Logit max diff:     {comparison.get('logit_max_diff', 'N/A'):.4f}")
        print(f"  Logit mean diff:    {comparison.get('logit_mean_diff', 'N/A'):.4f}")
        print(f"  Token match rate:   {comparison.get('token_match_rate', 0) * 100:.1f}%")
        print(f"  All outputs same:   {comparison.get('all_outputs_identical', False)}")
        print(f"  Equivalent:         {comparison.get('equivalent', False)}")

        # Show sample outputs
        print("\n  Sample outputs:")
        for sample in comparison.get("output_samples", [])[:2]:
            print(f"    Prompt: {sample['prompt']}")
            print(f"    Baseline: {sample['output_a'][:80]}...")
            print(f"    Compiled: {sample['output_b'][:80]}...")
            print(f"    Identical: {sample['identical']}")
            print()
    else:
        print(f"  Status: {comparison.get('status', 'unknown')}")
        print(f"  Error: {comparison.get('error', 'N/A')}")

    # Storage-size proof summary
    print("\nStorage-Size Proof:")
    print(f"  Baseline: {baseline_storage_proof.get('total_storage_mb', 'N/A')} MB "
          f"(views: {baseline_storage_proof.get('has_views', 'N/A')})")
    print(f"  Compiled: {compiled_storage_proof.get('total_storage_mb', 'N/A')} MB "
          f"(views: {compiled_storage_proof.get('has_views', 'N/A')})")

    save_results(results, output_dir / "numerical_check.json")

    # Cleanup weight directories
    shutil.rmtree(baseline_path, ignore_errors=True)
    shutil.rmtree(compiled_path, ignore_errors=True)

    return results


def run_memory_sentinel_check(output_dir: Path) -> dict:
    """
    Run memory sentinel validation to verify GPU memory measurement accuracy.

    This test allocates a known-size tensor and verifies the memory tracking
    is accurate within tolerance. Useful for validating that memory metrics
    in benchmark results are trustworthy.
    """
    print_header("MEMORY SENTINEL VALIDATION")

    print("Testing GPU memory measurement accuracy...")
    print("Allocating 1GB test tensor and verifying measurement...\n")

    result = run_memory_sentinel(
        allocation_gb=1.0,
        tolerance_mb=50.0,
    )

    print(f"Results:")
    if result.get("status") == "success":
        print(f"  Expected allocation: {result['expected_mb']:.2f} MB")
        print(f"  Measured allocation: {result['measured_mb']:.2f} MB")
        print(f"  Delta:              {result['delta_mb']:.2f} MB")
        print(f"  Within tolerance:   {result['within_tolerance']} (±{result['tolerance_mb']} MB)")
        print(f"  Freed correctly:    {result['freed_correctly']}")

        if result['within_tolerance'] and result['freed_correctly']:
            print("\n✅ Memory measurement is accurate - benchmark results are trustworthy")
        else:
            print("\n⚠️ Memory measurement may be inaccurate - interpret results with caution")
    else:
        print(f"  Status: {result.get('status')}")
        print(f"  Reason: {result.get('reason', 'unknown')}")

    save_results(result, output_dir / "memory_sentinel.json")
    return result


def run_both(scenario: str, output_dir: Path, num_runs: int = 3) -> dict:
    """Run both baseline and compiled benchmarks."""
    baseline_results = run_baseline(scenario, output_dir, num_runs)
    compiled_results = run_compiled(scenario, output_dir, num_runs)

    # Print comparison
    print_header("COMPARISON SUMMARY")

    if baseline_results.get("mean_time") and compiled_results.get("warm_mean"):
        baseline_time = baseline_results["mean_time"]
        compiled_warm = compiled_results["warm_mean"]
        compile_overhead = compiled_results.get("compile_overhead", 0)
        speedup = baseline_time / compiled_warm if compiled_warm > 0 else 0

        print(f"Baseline avg:      {format_time(baseline_time)}")
        print(f"Compiled cold:     {format_time(compiled_results['cold_time'])}")
        print(f"Compiled warm avg: {format_time(compiled_warm)}")
        print(f"Compile overhead:  {format_time(compile_overhead)}")
        print(f"Speedup:           {speedup:.2f}x")

        # Variance stats
        baseline_var = baseline_results.get("variance_stats", {})
        compiled_var = compiled_results.get("variance_stats", {})
        print(f"\nBaseline stddev:   {baseline_var.get('stddev', 0):.1f}s (CV: {baseline_var.get('cv', 0):.1f}%)")
        print(f"Compiled stddev:   {compiled_var.get('stddev', 0):.1f}s (CV: {compiled_var.get('cv', 0):.1f}%)")

        # Compile stats
        compile_stats = compiled_results.get("compile_stats", {})
        print(f"\nCompile stats:")
        print(f"  Graphs compiled: {compile_stats.get('num_graphs', 0)}")
        print(f"  Graph breaks:    {compile_stats.get('graph_breaks', 0)}")
        print(f"  Frames compiled: {compile_stats.get('frames_compiled', 0)}")

        # Memory comparison
        baseline_mem = baseline_results.get("peak_memory", {}).get("max_allocated_gb", 0)
        compiled_mem = compiled_results.get("peak_memory", {}).get("max_allocated_gb", 0)
        print(f"\nBaseline peak GPU: {baseline_mem:.2f} GB")
        print(f"Compiled peak GPU: {compiled_mem:.2f} GB")
        print(f"Memory overhead:   {((compiled_mem / baseline_mem) - 1) * 100:.1f}%" if baseline_mem > 0 else "N/A")

    return {
        "baseline": baseline_results,
        "compiled": compiled_results,
    }


def main():
    parser = argparse.ArgumentParser(description="GPTQ torch.compile Benchmark")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=list(SCENARIOS.keys()),
        required=True,
        help="Benchmark scenario to run",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "compiled", "both", "numerical_check", "memory_sentinel"],
        default="both",
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of measured runs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts",
        help="Directory for output files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.scenario
    output_dir.mkdir(parents=True, exist_ok=True)

    # Capture and save environment
    env_info = get_environment_info()
    save_results(env_info, output_dir / "env_info.json")

    print_header("GPTQ torch.compile Benchmark")
    print(f"Scenario: {args.scenario}")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")
    print(f"\nEnvironment:")
    print(f"  PyTorch: {env_info['torch_version']}")
    print(f"  CUDA: {env_info['torch_cuda_version']}")
    if env_info.get("gpu_name"):
        print(f"  GPU: {env_info['gpu_name']} ({env_info['gpu_vram_gb']} GB)")

    # Run benchmark
    if args.mode == "baseline":
        results = run_baseline(args.scenario, output_dir, args.num_runs)
    elif args.mode == "compiled":
        results = run_compiled(args.scenario, output_dir, args.num_runs)
    elif args.mode == "both":
        results = run_both(args.scenario, output_dir, args.num_runs)
    elif args.mode == "numerical_check":
        results = run_numerical_check(args.scenario, output_dir)
    elif args.mode == "memory_sentinel":
        results = run_memory_sentinel_check(output_dir)

    # Generate summary
    if args.mode == "both":
        print_header("MARKDOWN TABLE")
        table = generate_markdown_table(results)
        print(table)

        # Save summary
        summary_path = output_dir / "summary.md"
        with open(summary_path, "w") as f:
            f.write(f"# {args.scenario} Benchmark Results\n\n")
            f.write(f"## Environment\n")
            f.write(f"- PyTorch: {env_info['torch_version']}\n")
            f.write(f"- GPU: {env_info.get('gpu_name', 'N/A')}\n")
            f.write(f"- Git: {env_info.get('git_sha', 'N/A')[:8]}\n\n")
            f.write(f"## Results\n\n")
            f.write(table)
            f.write("\n\n")

            # Add variance stats section
            f.write("## Variance Statistics\n\n")
            baseline_var = results.get("baseline", {}).get("variance_stats", {})
            compiled_var = results.get("compiled", {}).get("variance_stats", {})
            f.write(f"| Mode | Mean | Stddev | CV | Min | Max |\n")
            f.write(f"|------|------|--------|----|----|-----|\n")
            if baseline_var:
                f.write(f"| baseline | {format_time(baseline_var.get('mean', 0))} | "
                        f"{baseline_var.get('stddev', 0):.1f}s | {baseline_var.get('cv', 0):.1f}% | "
                        f"{format_time(baseline_var.get('min', 0))} | {format_time(baseline_var.get('max', 0))} |\n")
            if compiled_var:
                f.write(f"| compiled | {format_time(compiled_var.get('mean', 0))} | "
                        f"{compiled_var.get('stddev', 0):.1f}s | {compiled_var.get('cv', 0):.1f}% | "
                        f"{format_time(compiled_var.get('min', 0))} | {format_time(compiled_var.get('max', 0))} |\n")
            f.write("\n")

            # Add compile stats section
            compile_stats = results.get("compiled", {}).get("compile_stats", {})
            f.write("## Compile Statistics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Graphs compiled | {compile_stats.get('num_graphs', 0)} |\n")
            f.write(f"| Graph breaks | {compile_stats.get('graph_breaks', 0)} |\n")
            f.write(f"| Frames compiled | {compile_stats.get('frames_compiled', 0)} |\n")
            f.write("\n")

            # For blocksize scenarios, add stress test conclusion
            if "blocksize" in args.scenario:
                conclusion = generate_stress_test_conclusion(results)
                f.write(conclusion)
                f.write("\n")

        print(f"\nSummary saved to: {summary_path}")

    print("\nBenchmark complete!")
    return 0


if __name__ == "__main__":
    exit(main())
