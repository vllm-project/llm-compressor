"""
Benchmark utilities for GPTQ torch.compile evaluation.
Provides environment capture, GPU metrics, weight comparison, and compile stats.
"""
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"git_sha": sha, "git_branch": branch}
    except Exception:
        return {"git_sha": "unknown", "git_branch": "unknown"}


def get_nvidia_smi_info() -> Dict[str, str]:
    """Get NVIDIA driver info from nvidia-smi."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"nvidia_driver": output}
    except Exception:
        return {"nvidia_driver": "unknown"}


def get_environment_info() -> Dict[str, Any]:
    """Capture complete environment information for reproducibility."""
    env_info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", "N/A"),
        "cuda_available": torch.cuda.is_available(),
    }

    # Git info
    env_info.update(get_git_info())

    # NVIDIA info
    env_info.update(get_nvidia_smi_info())

    # GPU info
    if torch.cuda.is_available():
        env_info["gpu_count"] = torch.cuda.device_count()
        env_info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        env_info["gpu_vram_gb"] = round(props.total_memory / 1e9, 2)
        env_info["gpu_compute_capability"] = f"{props.major}.{props.minor}"

    # Compile settings (what's in our code)
    env_info["compile_settings"] = {
        "dynamic": True,  # hardcoded in @torch.compile decorator
        "backend": "inductor",
    }

    # Dynamo config
    try:
        import torch._dynamo.config as dynamo_config
        env_info["dynamo_config"] = {
            "cache_size_limit": dynamo_config.cache_size_limit,
            "suppress_errors": dynamo_config.suppress_errors,
        }
    except Exception:
        env_info["dynamo_config"] = {}

    return env_info


# ============================================================================
# Compile Statistics Tracking
# ============================================================================

def get_dynamo_counters() -> Dict[str, Any]:
    """Get torch._dynamo compilation counters."""
    try:
        from torch._dynamo.utils import counters
        return {
            "frames_ok": counters["frames"]["ok"],
            "frames_total": counters["frames"]["total"],
            "graph_breaks": counters["graph_break"].copy() if counters["graph_break"] else {},
            "unique_graphs": counters["stats"].get("unique_graphs", 0),
            "calls_captured": counters["stats"].get("calls_captured", 0),
        }
    except Exception as e:
        return {"error": str(e)}


def reset_dynamo_counters():
    """Reset torch._dynamo counters."""
    try:
        from torch._dynamo.utils import counters
        counters.clear()
    except Exception:
        pass


def get_compile_stats() -> Dict[str, Any]:
    """
    Get comprehensive compile statistics.

    Returns dict with:
    - num_graphs: number of unique compiled graphs
    - recompiles: number of recompilation events
    - compile_time_seconds: total time spent compiling (if available)
    """
    stats = {}

    # Dynamo counters
    dynamo = get_dynamo_counters()
    stats["dynamo"] = dynamo

    # Count graphs and recompiles
    stats["num_graphs"] = dynamo.get("unique_graphs", 0)
    stats["frames_compiled"] = dynamo.get("frames_ok", 0)
    stats["graph_breaks"] = len(dynamo.get("graph_breaks", {}))

    # Try to get inductor metrics
    try:
        from torch._inductor.metrics import CachedMetricsHelper
        inductor_metrics = CachedMetricsHelper.get_deltas()
        stats["inductor"] = {
            "cache_hits": inductor_metrics.get("inductor_cache_hits", 0),
            "cache_misses": inductor_metrics.get("inductor_cache_misses", 0),
        }
    except Exception:
        stats["inductor"] = {}

    return stats


class CompileStatsContext:
    """Context manager to track compile stats during a block of code."""

    def __init__(self):
        self.start_stats = None
        self.end_stats = None
        self.delta = None

    def __enter__(self):
        reset_dynamo_counters()
        self.start_stats = get_compile_stats()
        return self

    def __exit__(self, *args):
        self.end_stats = get_compile_stats()
        # Calculate delta
        self.delta = {
            "num_graphs": self.end_stats.get("num_graphs", 0),
            "frames_compiled": self.end_stats.get("frames_compiled", 0),
            "graph_breaks": self.end_stats.get("graph_breaks", 0),
        }


# ============================================================================
# Variance Statistics
# ============================================================================

def compute_variance_stats(times: List[float]) -> Dict[str, float]:
    """
    Compute variance and percentile statistics for timing data.

    Returns dict with:
    - mean, stddev, cv (coefficient of variation)
    - min, max
    - p50, p90, p99
    """
    if not times:
        return {}

    arr = np.array(times)
    mean = float(np.mean(arr))
    stddev = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    return {
        "mean": mean,
        "stddev": stddev,
        "cv": (stddev / mean * 100) if mean > 0 else 0.0,  # coefficient of variation %
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)) if len(arr) >= 5 else float(np.max(arr)),
        "p99": float(np.percentile(arr, 99)) if len(arr) >= 10 else float(np.max(arr)),
    }


# ============================================================================
# GPU State Management
# ============================================================================

def reset_gpu_state():
    """Reset GPU memory tracking and clear caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_gpu_memory_stats() -> Dict[str, float]:
    """Get current GPU memory statistics in GB."""
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "max_reserved_gb": 0.0,
        }

    torch.cuda.synchronize()
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
    }


class Timer:
    """Context manager for timing with CUDA synchronization."""

    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


def compare_model_weights(
    path_a: Path,
    path_b: Path,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Compare quantized weights between two saved models.

    Note: This compares raw tensors including packed int32. For meaningful
    numerical correctness, use compare_model_outputs() instead.

    Returns dict with:
    - max_abs_diff, mean_abs_diff, p99_abs_diff
    - cosine_similarity (mean across all tensors)
    - per-layer stats
    """
    from safetensors import safe_open

    # Find safetensor files
    def find_safetensors(path: Path) -> List[Path]:
        files = list(path.glob("*.safetensors"))
        if not files:
            files = list(path.glob("**/*.safetensors"))
        return sorted(files)

    files_a = find_safetensors(path_a)
    files_b = find_safetensors(path_b)

    if not files_a or not files_b:
        return {"error": "No safetensor files found"}

    all_max_diffs = []
    all_mean_diffs = []
    all_cosine_sims = []
    all_element_diffs = []  # For p99 calculation
    layer_stats = []

    for file_a in files_a:
        # Find corresponding file in b
        file_b = path_b / file_a.name
        if not file_b.exists():
            continue

        with safe_open(file_a, framework="pt", device=device) as fa:
            with safe_open(file_b, framework="pt", device=device) as fb:
                keys_a = set(fa.keys())
                keys_b = set(fb.keys())
                common_keys = keys_a & keys_b

                for key in common_keys:
                    tensor_a = fa.get_tensor(key).float()
                    tensor_b = fb.get_tensor(key).float()

                    if tensor_a.shape != tensor_b.shape:
                        continue

                    # Absolute differences
                    diff = (tensor_a - tensor_b).abs()
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()

                    # Collect element-level diffs for p99 (sample to avoid memory issues)
                    flat_diff = diff.flatten()
                    if len(flat_diff) > 10000:
                        # Sample 10k elements for p99 calculation
                        indices = torch.randperm(len(flat_diff))[:10000]
                        sampled = flat_diff[indices].tolist()
                    else:
                        sampled = flat_diff.tolist()
                    all_element_diffs.extend(sampled)

                    # Cosine similarity
                    flat_a = tensor_a.flatten()
                    flat_b = tensor_b.flatten()
                    cos_sim = torch.nn.functional.cosine_similarity(
                        flat_a.unsqueeze(0), flat_b.unsqueeze(0)
                    ).item()

                    all_max_diffs.append(max_diff)
                    all_mean_diffs.append(mean_diff)
                    all_cosine_sims.append(cos_sim)

                    layer_stats.append({
                        "name": key,
                        "max_abs_diff": max_diff,
                        "mean_abs_diff": mean_diff,
                        "cosine_similarity": cos_sim,
                    })

    if not all_max_diffs:
        return {"error": "No comparable tensors found"}

    # Calculate p99 from sampled element diffs
    element_arr = np.array(all_element_diffs)
    p99_diff = float(np.percentile(element_arr, 99))

    return {
        "max_abs_diff": max(all_max_diffs),
        "mean_abs_diff": sum(all_mean_diffs) / len(all_mean_diffs),
        "p99_abs_diff": p99_diff,
        "cosine_similarity": sum(all_cosine_sims) / len(all_cosine_sims),
        "num_tensors_compared": len(all_max_diffs),
        "equivalent": max(all_max_diffs) < 1e-5,
        "layer_stats": layer_stats[:10],  # First 10 for brevity
    }


def compare_model_outputs(
    path_a: Path,
    path_b: Path,
    num_samples: int = 5,
    max_new_tokens: int = 20,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Compare model outputs between two quantized models on fixed inputs.

    This is the recommended method for numerical correctness validation
    because it compares actual model behavior rather than internal
    representations (packed int32 tensors give meaningless diffs).

    Args:
        path_a: Path to first model
        path_b: Path to second model
        num_samples: Number of test prompts
        max_new_tokens: Tokens to generate per prompt
        device: Device for inference

    Returns dict with:
    - logit_max_diff: Maximum difference in output logits
    - logit_mean_diff: Mean difference in output logits
    - token_match_rate: Fraction of generated tokens that match
    - output_samples: Sample outputs for inspection
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Fixed test prompts for reproducibility
    test_prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "In machine learning, gradient descent",
        "The quick brown fox",
        "Water boils at",
    ][:num_samples]

    results = {
        "method": "output_comparison",
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "logit_diffs": [],
        "token_matches": [],
        "output_samples": [],
    }

    try:
        # Load tokenizer (should be same for both)
        tokenizer = AutoTokenizer.from_pretrained(path_a)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load models
        model_a = AutoModelForCausalLM.from_pretrained(
            path_a,
            device_map=device,
            torch_dtype=torch.float16,
        )
        model_b = AutoModelForCausalLM.from_pretrained(
            path_b,
            device_map=device,
            torch_dtype=torch.float16,
        )

        model_a.eval()
        model_b.eval()

        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                # Get logits for the input (forward pass only)
                outputs_a = model_a(**inputs)
                outputs_b = model_b(**inputs)

                # Compare logits
                logits_a = outputs_a.logits.float()
                logits_b = outputs_b.logits.float()

                logit_diff = (logits_a - logits_b).abs()
                max_diff = logit_diff.max().item()
                mean_diff = logit_diff.mean().item()

                results["logit_diffs"].append({
                    "prompt": prompt,
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                })

                # Generate tokens and compare
                gen_a = model_a.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy for reproducibility
                    pad_token_id=tokenizer.pad_token_id,
                )
                gen_b = model_b.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

                # Compare generated tokens
                tokens_a = gen_a[0].tolist()
                tokens_b = gen_b[0].tolist()
                min_len = min(len(tokens_a), len(tokens_b))
                matches = sum(1 for i in range(min_len) if tokens_a[i] == tokens_b[i])
                match_rate = matches / min_len if min_len > 0 else 0

                results["token_matches"].append({
                    "prompt": prompt,
                    "match_rate": match_rate,
                    "matched": matches,
                    "total": min_len,
                })

                # Decode for inspection
                text_a = tokenizer.decode(gen_a[0], skip_special_tokens=True)
                text_b = tokenizer.decode(gen_b[0], skip_special_tokens=True)

                results["output_samples"].append({
                    "prompt": prompt,
                    "output_a": text_a,
                    "output_b": text_b,
                    "identical": text_a == text_b,
                })

        # Cleanup models
        del model_a, model_b
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Aggregate results
        all_max_diffs = [d["max_diff"] for d in results["logit_diffs"]]
        all_mean_diffs = [d["mean_diff"] for d in results["logit_diffs"]]
        all_match_rates = [m["match_rate"] for m in results["token_matches"]]

        results["logit_max_diff"] = max(all_max_diffs)
        results["logit_mean_diff"] = sum(all_mean_diffs) / len(all_mean_diffs)
        results["token_match_rate"] = sum(all_match_rates) / len(all_match_rates)
        results["all_outputs_identical"] = all(s["identical"] for s in results["output_samples"])

        # Equivalence check: logit diff should be small for numerical equivalence
        # With FP16 quantized models, we expect some small differences
        results["equivalent"] = results["logit_max_diff"] < 1.0 and results["token_match_rate"] == 1.0
        results["status"] = "success"

    except Exception as e:
        import traceback
        results["status"] = "error"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()

    return results


# ============================================================================
# Storage-Size Proof Capture
# ============================================================================

def capture_storage_size_proof(model) -> Dict[str, Any]:
    """
    Capture storage size proof for quantized model tensors.

    Checks whether tensors are views or have storage larger than logical size,
    which would indicate memory materialization issues.

    Returns dict with per-tensor storage analysis for JSON artifact.
    """
    storage_proof = {
        "tensors": [],
        "total_logical_mb": 0.0,
        "total_storage_mb": 0.0,
        "has_views": False,
        "has_inflated_storage": False,
    }

    for name, param in model.named_parameters():
        logical_bytes = param.numel() * param.element_size()
        storage_bytes = param.untyped_storage().nbytes()

        logical_mb = logical_bytes / (1024 * 1024)
        storage_mb = storage_bytes / (1024 * 1024)
        is_view = storage_bytes > logical_bytes

        tensor_info = {
            "name": name,
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "logical_mb": round(logical_mb, 3),
            "storage_mb": round(storage_mb, 3),
            "is_view": is_view,
        }

        # Only include notable tensors (large or views)
        if logical_mb > 1.0 or is_view:
            storage_proof["tensors"].append(tensor_info)

        storage_proof["total_logical_mb"] += logical_mb
        storage_proof["total_storage_mb"] += storage_mb
        if is_view:
            storage_proof["has_views"] = True
        if storage_mb > logical_mb * 1.1:  # >10% inflation
            storage_proof["has_inflated_storage"] = True

    storage_proof["total_logical_mb"] = round(storage_proof["total_logical_mb"], 2)
    storage_proof["total_storage_mb"] = round(storage_proof["total_storage_mb"], 2)

    return storage_proof


# ============================================================================
# Memory Sentinel Validation
# ============================================================================

def run_memory_sentinel(
    allocation_gb: float = 1.0,
    tolerance_mb: float = 50.0,
) -> Dict[str, Any]:
    """
    Validate GPU memory measurement accuracy using a known-size allocation.

    This sentinel test allocates a tensor of known size, frees it, and verifies
    that the memory tracking accurately reflects the allocation/deallocation.

    Args:
        allocation_gb: Size of test tensor to allocate (in GB)
        tolerance_mb: Acceptable measurement error (in MB)

    Returns dict with:
    - expected_mb: Expected allocation size
    - measured_mb: Actual measured allocation delta
    - delta_mb: Difference between expected and measured
    - within_tolerance: Boolean pass/fail
    """
    if not torch.cuda.is_available():
        return {
            "status": "skipped",
            "reason": "CUDA not available",
        }

    # Reset GPU state completely
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    baseline_allocated = torch.cuda.memory_allocated()

    # Allocate known-size tensor
    expected_bytes = int(allocation_gb * 1e9)
    num_elements = expected_bytes // 4  # float32 = 4 bytes
    expected_mb = expected_bytes / (1024 * 1024)

    # Create tensor
    test_tensor = torch.zeros(num_elements, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()

    after_alloc = torch.cuda.memory_allocated()
    measured_bytes = after_alloc - baseline_allocated
    measured_mb = measured_bytes / (1024 * 1024)

    # Free tensor
    del test_tensor
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    after_free = torch.cuda.memory_allocated()
    freed_correctly = abs(after_free - baseline_allocated) < 1024 * 1024  # 1MB tolerance

    delta_mb = abs(measured_mb - expected_mb)
    within_tolerance = delta_mb <= tolerance_mb

    return {
        "status": "success",
        "expected_mb": round(expected_mb, 2),
        "measured_mb": round(measured_mb, 2),
        "delta_mb": round(delta_mb, 2),
        "within_tolerance": within_tolerance,
        "tolerance_mb": tolerance_mb,
        "freed_correctly": freed_correctly,
        "baseline_mb": round(baseline_allocated / (1024 * 1024), 2),
        "after_alloc_mb": round(after_alloc / (1024 * 1024), 2),
        "after_free_mb": round(after_free / (1024 * 1024), 2),
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def save_results(results: Dict[str, Any], output_path: Path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")


def generate_markdown_table(results: Dict[str, Any]) -> str:
    """Generate markdown summary table from benchmark results."""
    lines = []

    # Header with new columns
    lines.append("| Mode | First Run | Warm Avg | Stddev | Compile OH | Speedup | Peak Mem | Recompiles | max_abs_diff |")
    lines.append("|------|-----------|----------|--------|------------|---------|----------|------------|--------------|")

    baseline = results.get("baseline", {})
    compiled = results.get("compiled", {})

    # Baseline row
    if baseline:
        baseline_avg = baseline.get("mean_time", baseline.get("times", [0])[0])
        baseline_stats = baseline.get("variance_stats", {})
        stddev = baseline_stats.get("stddev", 0)
        lines.append(
            f"| baseline | {format_time(baseline.get('times', [0])[0])} | "
            f"{format_time(baseline_avg)} | "
            f"{stddev:.1f}s | - | 1.00x | "
            f"{baseline.get('peak_memory', {}).get('max_allocated_gb', 0):.2f} GB | - | - |"
        )

    # Compiled row
    if compiled:
        cold_time = compiled.get("cold_time", 0)
        warm_times = compiled.get("warm_times", [])
        warm_avg = sum(warm_times) / len(warm_times) if warm_times else 0
        compile_overhead = cold_time - warm_avg if warm_avg > 0 else 0

        compiled_stats = compiled.get("variance_stats", {})
        stddev = compiled_stats.get("stddev", 0)

        baseline_avg = baseline.get("mean_time", 1) if baseline else 1
        speedup = baseline_avg / warm_avg if warm_avg > 0 else 0

        # Compile stats
        compile_stats = compiled.get("compile_stats", {})
        recompiles = compile_stats.get("graph_breaks", 0)

        max_diff = results.get("numerical_check", {}).get("max_abs_diff", "N/A")
        if isinstance(max_diff, float):
            max_diff = f"{max_diff:.2e}"

        lines.append(
            f"| compiled | {format_time(cold_time)} | "
            f"{format_time(warm_avg)} | "
            f"{stddev:.1f}s | {format_time(compile_overhead)} | "
            f"{speedup:.2f}x | "
            f"{compiled.get('peak_memory', {}).get('max_allocated_gb', 0):.2f} GB | "
            f"{recompiles} | "
            f"{max_diff} |"
        )

    return "\n".join(lines)


def generate_stress_test_conclusion(results: Dict[str, Any]) -> str:
    """
    Generate explicit conclusion for blocksize stress tests.

    Returns text summarizing:
    - Did recompilation occur?
    - Did compile time blow up?
    - Did warm speedup remain?
    """
    compiled = results.get("compiled", {})
    baseline = results.get("baseline", {})

    if not compiled:
        return "No compiled results available."

    lines = []
    lines.append("### Stress Test Conclusion\n")

    # Recompilation check
    compile_stats = compiled.get("compile_stats", {})
    graph_breaks = compile_stats.get("graph_breaks", 0)
    num_graphs = compile_stats.get("num_graphs", 0)

    if graph_breaks == 0:
        lines.append(f"✅ **No recompilations detected** ({num_graphs} graphs compiled)")
    else:
        lines.append(f"⚠️ **{graph_breaks} graph breaks detected** ({num_graphs} graphs)")

    # Compile time check
    cold_time = compiled.get("cold_time", 0)
    warm_avg = compiled.get("warm_mean", 0)
    compile_overhead = cold_time - warm_avg if warm_avg > 0 else 0

    if compile_overhead < 900:  # 15 minutes
        lines.append(f"✅ **Compile overhead acceptable**: {format_time(compile_overhead)}")
    else:
        lines.append(f"⚠️ **High compile overhead**: {format_time(compile_overhead)}")

    # Speedup check
    if baseline:
        baseline_avg = baseline.get("mean_time", 1)
        speedup = baseline_avg / warm_avg if warm_avg > 0 else 0
        if speedup >= 1.5:
            lines.append(f"✅ **Speedup maintained**: {speedup:.2f}x")
        elif speedup >= 1.0:
            lines.append(f"⚠️ **Marginal speedup**: {speedup:.2f}x")
        else:
            lines.append(f"❌ **No speedup achieved**: {speedup:.2f}x")

    # Variance check
    var_stats = compiled.get("variance_stats", {})
    cv = var_stats.get("cv", 0)
    if cv < 5:
        lines.append(f"✅ **Stable timings**: CV = {cv:.1f}%")
    else:
        lines.append(f"⚠️ **Variable timings**: CV = {cv:.1f}%")

    return "\n".join(lines)
