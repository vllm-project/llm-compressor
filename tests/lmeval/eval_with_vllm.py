#!/usr/bin/env python3
"""
Evaluate quantized models using vLLM backend with lm-eval-harness.

This script provides an equivalent evaluation to test_lmeval.py but using
vLLM as the inference backend instead of HuggingFace transformers.

Usage:
    python eval_with_vllm.py --config tests/lmeval/configs/w4a16_awq_sym.yaml
    python eval_with_vllm.py --model /path/to/quantized/model --task gsm8k
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from loguru import logger
from pydantic import BaseModel

try:
    import lm_eval
    from lm_eval.models.vllm_vlms import VLLM

    lm_eval_installed = True
except ImportError:
    lm_eval_installed = False
    logger.error("lm_eval is not installed. Install with: pip install lm-eval")
    sys.exit(1)


class LmEvalConfig(BaseModel):
    """Configuration for lm-eval evaluation."""
    model: str = "vllm"
    model_args: dict = {}
    task: str = "gsm8k"
    num_fewshot: int = 5
    limit: int = 1000
    batch_size: int = 100
    apply_chat_template: bool = False
    recovery_threshold: dict | float = 0.95
    metrics: Optional[dict] = None


class VLLMEvaluator:
    """Evaluator using vLLM backend for lm-eval-harness."""

    def __init__(
        self,
        model_path: str,
        eval_config: LmEvalConfig,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
    ):
        """
        Initialize the vLLM evaluator.

        Args:
            model_path: Path to the model (can be HF model ID or local path)
            eval_config: Evaluation configuration
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type for inference (auto, float16, bfloat16)
        """
        self.model_path = model_path
        self.eval_config = eval_config
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype

    def evaluate(self) -> dict:
        """
        Run evaluation using vLLM backend.

        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating model: {self.model_path}")
        logger.info(f"Task: {self.eval_config.task}")
        logger.info(f"Num fewshot: {self.eval_config.num_fewshot}")
        logger.info(f"Limit: {self.eval_config.limit}")
        logger.info(f"GPU memory utilization: {self.gpu_memory_utilization}")

        # Create vLLM model instance
        vllm_model = VLLM(
            pretrained=self.model_path,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=None,  # Auto-detect
            **self.eval_config.model_args,
        )

        # Run evaluation
        logger.info("Starting evaluation...")
        results = lm_eval.simple_evaluate(
            model=vllm_model,
            tasks=[self.eval_config.task],
            num_fewshot=self.eval_config.num_fewshot,
            limit=self.eval_config.limit,
            apply_chat_template=self.eval_config.apply_chat_template,
            batch_size=self.eval_config.batch_size,
        )

        return results

    def print_results(self, results: dict) -> None:
        """Print evaluation results in a readable format."""
        logger.info("=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)

        task_results = results["results"][self.eval_config.task]

        for metric_key, value in task_results.items():
            if "stderr" not in metric_key and not metric_key.startswith("alias"):
                logger.info(f"{metric_key:40} | {value:.4f}")

        logger.info("=" * 80)

    def compare_with_baseline(
        self,
        baseline_results: dict,
        compressed_results: dict
    ) -> None:
        """
        Compare compressed model results with baseline.

        Args:
            baseline_results: Results from baseline/unquantized model
            compressed_results: Results from compressed/quantized model
        """
        baseline_metrics = baseline_results["results"][self.eval_config.task]
        compressed_metrics = compressed_results["results"][self.eval_config.task]
        higher_is_better_map = compressed_results.get("higher_is_better", {}).get(
            self.eval_config.task, {}
        )

        logger.info("=" * 80)
        logger.info("RECOVERY COMPARISON")
        logger.info("=" * 80)

        default_threshold = 0.95
        if isinstance(self.eval_config.recovery_threshold, dict):
            default_threshold = 0.95
        else:
            default_threshold = self.eval_config.recovery_threshold

        failures = []

        for metric_key, compressed_val in compressed_metrics.items():
            if "stderr" in metric_key or metric_key.startswith("alias"):
                continue

            baseline_val = baseline_metrics.get(metric_key)
            if baseline_val is None:
                logger.warning(f"Metric {metric_key} not found in baseline results")
                continue

            # Get threshold
            if isinstance(self.eval_config.recovery_threshold, dict):
                threshold = self.eval_config.recovery_threshold.get(
                    metric_key, default_threshold
                )
            else:
                threshold = self.eval_config.recovery_threshold

            # Get direction
            base_metric_name = metric_key.split(",")[0]
            higher_is_better = higher_is_better_map.get(base_metric_name, True)

            # Compute recovery
            if baseline_val == 0:
                recovery = 1.0 if compressed_val == 0 else 0.0
            elif higher_is_better:
                recovery = compressed_val / baseline_val
            else:
                recovery = baseline_val / compressed_val

            # Round to nearest percent
            recovery = (torch.round(torch.tensor(recovery) * 100) / 100).item()
            passed = recovery >= threshold
            direction = "↑" if higher_is_better else "↓"

            msg = (
                f"{metric_key:40} | Baseline: {baseline_val:.4f} | "
                f"Compressed: {compressed_val:.4f} | "
                f"Recovery: {recovery:6.2%} {direction} | Threshold: ≥{threshold:.2%}"
            )

            if passed:
                logger.info(f"✓ {msg}")
            else:
                logger.error(f"✗ {msg}")
                failures.append(
                    f"{metric_key}: {recovery:.2%} < {threshold:.2%}"
                )

        logger.info("=" * 80)

        if failures:
            logger.error("FAILED: Recovery thresholds not met")
            for failure in failures:
                logger.error(f"  - {failure}")
            return False
        else:
            logger.info("✓ ALL METRICS PASSED RECOVERY THRESHOLDS")
            return True


def load_config_from_yaml(config_path: str) -> tuple[str, LmEvalConfig, Optional[str]]:
    """
    Load configuration from YAML file.

    Returns:
        Tuple of (model_id, eval_config, quantized_model_path)
    """
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    model = config["model"]
    scheme = config.get("scheme", "unknown")
    lmeval_config = LmEvalConfig(**config.get("lmeval", {}))

    # Assume quantized model is saved with this naming pattern
    # (matching the test_lmeval.py behavior)
    save_dir = config.get("save_dir")
    if not save_dir:
        save_dir = model.split("/")[1] + f"-{scheme}"

    return model, lmeval_config, save_dir


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate quantized models using vLLM backend"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (same format as lmeval configs)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model or HuggingFace model ID",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="gsm8k",
        help="Evaluation task (default: gsm8k)",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=5,
        help="Number of few-shot examples (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Limit number of evaluation examples (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for evaluation (default: 100)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for inference (default: auto)",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also evaluate baseline model and compare",
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        base_model, eval_config, quantized_model = load_config_from_yaml(args.config)

        # If --model provided, use that; otherwise use quantized_model from config
        if args.model:
            model_to_eval = args.model
        elif os.path.exists(quantized_model):
            model_to_eval = quantized_model
        else:
            logger.error(
                f"Quantized model not found at {quantized_model}. "
                "Please run quantization first:\n"
                f"  python tests/lmeval/quantize_for_lmeval.py --config {args.config}"
            )
            sys.exit(1)
    else:
        if not args.model:
            logger.error("Either --config or --model must be provided")
            sys.exit(1)

        model_to_eval = args.model
        base_model = None
        eval_config = LmEvalConfig(
            task=args.task,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            batch_size=args.batch_size,
        )

    # Evaluate quantized/compressed model
    logger.info("=" * 80)
    logger.info("EVALUATING MODEL WITH VLLM")
    logger.info("=" * 80)

    evaluator = VLLMEvaluator(
        model_path=model_to_eval,
        eval_config=eval_config,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )

    compressed_results = evaluator.evaluate()
    evaluator.print_results(compressed_results)

    # Compare with baseline if requested
    if args.compare_baseline and base_model:
        logger.info("=" * 80)
        logger.info("EVALUATING BASELINE MODEL")
        logger.info("=" * 80)

        baseline_evaluator = VLLMEvaluator(
            model_path=base_model,
            eval_config=eval_config,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
        )

        baseline_results = baseline_evaluator.evaluate()
        baseline_evaluator.print_results(baseline_results)

        # Compare
        success = evaluator.compare_with_baseline(baseline_results, compressed_results)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
