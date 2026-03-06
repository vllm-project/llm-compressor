#!/usr/bin/env python3
"""
Small lm_eval test script: compare AWQ with vs without smooth_layer_quantization.

Measures AWQ compression runtime and lm_eval metrics for both configs.

Usage:
  # Without smooth layer quantization (baseline)
  python lm_eval_smooth_layer_test.py --without-smooth

  # With smooth layer quantization
  python lm_eval_smooth_layer_test.py --with-smooth

  # Run both and compare runtime
  python lm_eval_smooth_layer_test.py --both

"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048

# Recipe paths (relative to repo root)
RECIPE_WITHOUT_SMOOTH = "tests/e2e/vLLM/recipes/WNA16/recipe_w4a16_awq_sym.yaml"
RECIPE_WITH_SMOOTH = (
    "tests/e2e/vLLM/recipes/WNA16/recipe_w4a16_awq_sym_with_smooth.yaml"
)

# lm_eval settings
LMEVAL_TASK = "gsm8k"
LMEVAL_LIMIT = 100
LMEVAL_NUM_FEWSHOT = 5
LMEVAL_BATCH_SIZE = 16


def run_compression(recipe_path: str, save_dir: str) -> float:
    """Run AWQ compression and save model to save_dir. Returns elapsed time (s)."""
    from tests.e2e.e2e_utils import run_oneshot_for_e2e_testing

    recipe_abs = REPO_ROOT / recipe_path
    if not recipe_abs.exists():
        raise FileNotFoundError(f"Recipe not found: {recipe_abs}")

    with_smooth = "with_smooth" in recipe_path
    print(f"Loading model: {MODEL_ID}")
    print(f"Recipe: {recipe_path} (smooth_layer_quantization={with_smooth})")
    t0 = time.perf_counter()
    model, processor = run_oneshot_for_e2e_testing(
        model=MODEL_ID,
        model_class="AutoModelForCausalLM",
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        scheme="W4A16_awq_sym",
        dataset_id=DATASET_ID,
        dataset_config=None,
        dataset_split=DATASET_SPLIT,
        recipe=str(recipe_abs),
        quant_type=None,
    )
    print(f"Saving compressed model to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    # Reset session so a subsequent run (e.g. --both) gets a clean state
    from llmcompressor.core import active_session

    active_session().reset()
    elapsed = time.perf_counter() - t0
    print(f"Compression done in {elapsed:.2f}s.\n")
    return elapsed


def run_lm_eval(model_path: str) -> dict:
    """Run lm_eval on a model path; returns results dict."""
    try:
        import lm_eval
        import lm_eval.api.registry
        import lm_eval.models  # noqa: F401 - populate registry
    except ImportError:
        print("lm_eval not installed. Install with: pip install lm_eval==0.4.9.2")
        return {}

    from tests.e2e.e2e_utils import load_model

    lm_eval_cls = lm_eval.api.registry.get_model("hf")
    model = load_model(model_path, "AutoModelForCausalLM", device_map="cuda:0")
    results = lm_eval.simple_evaluate(
        model=lm_eval_cls(
            pretrained=model,
            batch_size=LMEVAL_BATCH_SIZE,
            add_bos_token=True,
            dtype="bfloat16",
        ),
        tasks=[LMEVAL_TASK],
        num_fewshot=LMEVAL_NUM_FEWSHOT,
        limit=LMEVAL_LIMIT,
        apply_chat_template=False,
        batch_size=LMEVAL_BATCH_SIZE,
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="AWQ lm_eval test: with vs without smooth_layer_quantization"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--without-smooth",
        action="store_true",
        help="Run baseline (no smooth layer quant)",
    )
    group.add_argument(
        "--with-smooth", action="store_true", help="Run with smooth_layer_quantization"
    )
    group.add_argument(
        "--both", action="store_true", help="Run both and compare metrics"
    )
    parser.add_argument(
        "--skip-lm-eval", action="store_true", help="Only run compression, skip lm_eval"
    )
    args = parser.parse_args()

    if not args.skip_lm_eval:
        try:
            import torch

            if not torch.cuda.is_available():
                print(
                    "Warning: No CUDA GPU. lm_eval will be skipped. "
                    "Use --skip-lm-eval to only run compression."
                )
                args.skip_lm_eval = True
        except Exception:
            args.skip_lm_eval = True

    os.chdir(REPO_ROOT)
    results_baseline = {}
    results_smooth = {}
    awq_time_baseline: float | None = None
    awq_time_smooth: float | None = None

    if args.without_smooth or args.both:
        save_baseline = "qwen3-0.6b-w4a16-awq-baseline"
        awq_time_baseline = run_compression(RECIPE_WITHOUT_SMOOTH, save_baseline)
        if not args.skip_lm_eval:
            print("Running lm_eval on baseline (without smooth)...")
            results_baseline = run_lm_eval(save_baseline)
            if results_baseline:
                m = results_baseline.get("results", {}).get(LMEVAL_TASK, {})
                print(f"Baseline metrics: {m}")

    if args.with_smooth or args.both:
        save_smooth = "qwen3-0.6b-w4a16-awq-with-smooth"
        awq_time_smooth = run_compression(RECIPE_WITH_SMOOTH, save_smooth)
        if not args.skip_lm_eval:
            print("Running lm_eval on model with smooth_layer_quantization...")
            results_smooth = run_lm_eval(save_smooth)
            if results_smooth:
                m = results_smooth.get("results", {}).get(LMEVAL_TASK, {})
                print(f"With-smooth metrics: {m}")

    # AWQ runtime comparison (when both configs were run)
    if args.both and awq_time_baseline is not None and awq_time_smooth is not None:
        print("\n========== AWQ RUNTIME COMPARISON ==========")
        print(f"  Baseline (no smooth_layer_quantization): {awq_time_baseline:.2f}s")
        print(f"  With smooth_layer_quantization:           {awq_time_smooth:.2f}s")
        diff = awq_time_smooth - awq_time_baseline
        pct = (diff / awq_time_baseline * 100) if awq_time_baseline else 0
        print(f"  Difference: {diff:+.2f}s ({pct:+.1f}%)")
        print("=============================================\n")

    if args.both and not args.skip_lm_eval and results_baseline and results_smooth:
        base_m = results_baseline.get("results", {}).get(LMEVAL_TASK, {})
        smooth_m = results_smooth.get("results", {}).get(LMEVAL_TASK, {})
        print("\n========== LMEVAL METRICS COMPARISON ==========")
        for k in set(base_m) | set(smooth_m):
            b = base_m.get(k, "N/A")
            s = smooth_m.get(k, "N/A")
            print(f"  {k}: baseline={b}  with_smooth={s}")
        print("===============================================\n")
    print("Done.")


if __name__ == "__main__":
    main()
