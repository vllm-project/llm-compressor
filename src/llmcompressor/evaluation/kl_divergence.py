"""
KL-divergence evaluation tool for comparing quantized models against
their base (unquantized) counterparts.

Computes forward KLD(base || quant) and reverse KLD(quant || base) to
measure how well a quantized model preserves the original probability
distribution.  KL divergence is asymmetric, so both directions are
reported.

Usage (CLI)::

    python -m llmcompressor.evaluation.kl_divergence \\
        --base-model meta-llama/Llama-3.1-8B \\
        --target-model quantized-model-path \\
        --dataset wikitext --dataset-config wikitext-2-raw-v1 \\
        --num-samples 128 --max-seq-length 512

Usage (Python API)::

    from llmcompressor.evaluation.kl_divergence import evaluate_kl_divergence

    results = evaluate_kl_divergence(
        base_model="meta-llama/Llama-3.1-8B",
        target_model="quantized-model-path",
        dataset_id="wikitext",
        dataset_config="wikitext-2-raw-v1",
        num_samples=128,
        max_seq_length=512,
    )
    print(results)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["evaluate_kl_divergence", "KLDivergenceResult"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class KLDivergenceResult:
    """Stores per-sample and aggregate KL-divergence statistics."""

    # Aggregate
    forward_kld_mean: float = 0.0  # KLD(base || target), averaged over tokens
    reverse_kld_mean: float = 0.0  # KLD(target || base), averaged over tokens
    symmetric_kld_mean: float = 0.0  # (forward + reverse) / 2
    num_samples: int = 0
    num_tokens: int = 0

    # Per-sample lists
    forward_kld_per_sample: list[float] = field(default_factory=list)
    reverse_kld_per_sample: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"KL-Divergence Evaluation ({self.num_samples} samples, "
            f"{self.num_tokens} tokens)\n"
            f"  KLD(base || target): {self.forward_kld_mean:.6f}\n"
            f"  KLD(target || base): {self.reverse_kld_mean:.6f}\n"
            f"  Symmetric KLD:       {self.symmetric_kld_mean:.6f}"
        )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _kl_divergence_per_token(
    log_probs_p: torch.Tensor,
    log_probs_q: torch.Tensor,
) -> torch.Tensor:
    """
    Compute token-level KL(P || Q) from log-probabilities.

    KL(P || Q) = sum_x P(x) * (log P(x) - log Q(x))

    :param log_probs_p: shape (seq_len, vocab_size), log-softmax of P
    :param log_probs_q: shape (seq_len, vocab_size), log-softmax of Q
    :return: shape (seq_len,), per-token KL divergence
    """
    p = log_probs_p.exp()
    kl = (p * (log_probs_p - log_probs_q)).sum(dim=-1)
    return kl


@torch.no_grad()
def _collect_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Run a forward pass and return logits (on CPU to save GPU memory).

    :param model: the causal LM
    :param input_ids: shape (1, seq_len)
    :param attention_mask: shape (1, seq_len) or None
    :return: logits tensor of shape (seq_len, vocab_size)
    """
    outputs = model(
        input_ids=input_ids.to(model.device),
        attention_mask=(
            attention_mask.to(model.device) if attention_mask is not None else None
        ),
    )
    # Move to CPU immediately to free GPU memory
    return outputs.logits[0].float().cpu()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_kl_divergence(
    base_model: str | torch.nn.Module,
    target_model: str | torch.nn.Module,
    dataset_id: str = "wikitext",
    dataset_config: str | None = "wikitext-2-raw-v1",
    dataset_split: str = "test",
    text_column: str = "text",
    num_samples: int = 128,
    max_seq_length: int = 512,
    batch_size: int = 1,
    device: str | None = None,
    base_model_kwargs: dict | None = None,
    target_model_kwargs: dict | None = None,
) -> KLDivergenceResult:
    """
    Evaluate KL divergence between a base model and a target (quantized) model.

    :param base_model: HuggingFace model ID or an already-loaded model
    :param target_model: HuggingFace model ID or an already-loaded model
    :param dataset_id: HuggingFace dataset ID for evaluation
    :param dataset_config: dataset configuration name (e.g. "wikitext-2-raw-v1")
    :param dataset_split: dataset split to use
    :param text_column: name of the text column in the dataset
    :param num_samples: number of samples to evaluate
    :param max_seq_length: maximum token sequence length
    :param batch_size: not used yet (reserved for future batched evaluation)
    :param device: device to run on ("cuda", "cpu", "auto"). Defaults to "auto"
    :param base_model_kwargs: additional kwargs for AutoModelForCausalLM.from_pretrained
    :param target_model_kwargs: additional kwargs for AutoModelForCausalLM.from_pretrained
    :return: KLDivergenceResult with per-sample and aggregate statistics
    """
    if device is None:
        device = "auto"

    base_model_kwargs = base_model_kwargs or {}
    target_model_kwargs = target_model_kwargs or {}

    # --- Load models ---
    if isinstance(base_model, str):
        logger.info("Loading base model: %s", base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map=device,
            **base_model_kwargs,
        )
        base_model_id = base_model
    else:
        base_model_obj = base_model
        base_model_id = getattr(base_model, "name_or_path", "base_model")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    if isinstance(target_model, str):
        logger.info("Loading target model: %s", target_model)
        target_model_obj = AutoModelForCausalLM.from_pretrained(
            target_model,
            torch_dtype="auto",
            device_map=device,
            **target_model_kwargs,
        )
    else:
        target_model_obj = target_model

    base_model_obj.eval()
    target_model_obj.eval()

    # --- Load and tokenize dataset ---
    logger.info(
        "Loading dataset: %s (config=%s, split=%s)",
        dataset_id,
        dataset_config,
        dataset_split,
    )

    ds_kwargs = {"split": dataset_split}
    if dataset_config:
        ds = load_dataset(dataset_id, dataset_config, **ds_kwargs)
    else:
        ds = load_dataset(dataset_id, **ds_kwargs)

    # Filter out empty texts
    ds = ds.filter(lambda x: len(x[text_column].strip()) > 0)

    if len(ds) > num_samples:
        ds = ds.select(range(num_samples))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Evaluate ---
    result = KLDivergenceResult(num_samples=len(ds))
    total_forward_kld = 0.0
    total_reverse_kld = 0.0
    total_tokens = 0

    for sample in tqdm(ds, desc="Evaluating KL divergence"):
        text = sample[text_column]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_seq_length,
            truncation=True,
            padding=False,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        if input_ids.shape[1] < 2:
            continue

        # Get logits from both models
        base_logits = _collect_logits(base_model_obj, input_ids, attention_mask)
        target_logits = _collect_logits(
            target_model_obj, input_ids, attention_mask
        )

        # Convert to log-probabilities
        base_log_probs = F.log_softmax(base_logits, dim=-1)
        target_log_probs = F.log_softmax(target_logits, dim=-1)

        # Compute KL divergence per token (skip the last position which has no
        # next-token prediction to compare against, but for logit comparison
        # all positions are valid)
        forward_kld = _kl_divergence_per_token(base_log_probs, target_log_probs)
        reverse_kld = _kl_divergence_per_token(target_log_probs, base_log_probs)

        # Clamp to avoid negative values from numerical imprecision
        forward_kld = forward_kld.clamp(min=0.0)
        reverse_kld = reverse_kld.clamp(min=0.0)

        seq_len = forward_kld.shape[0]
        sample_fwd = forward_kld.mean().item()
        sample_rev = reverse_kld.mean().item()

        result.forward_kld_per_sample.append(sample_fwd)
        result.reverse_kld_per_sample.append(sample_rev)

        total_forward_kld += forward_kld.sum().item()
        total_reverse_kld += reverse_kld.sum().item()
        total_tokens += seq_len

    # Aggregate
    if total_tokens > 0:
        result.forward_kld_mean = total_forward_kld / total_tokens
        result.reverse_kld_mean = total_reverse_kld / total_tokens
        result.symmetric_kld_mean = (
            result.forward_kld_mean + result.reverse_kld_mean
        ) / 2.0
    result.num_tokens = total_tokens

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KL divergence between a base model and a "
        "quantized/target model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="HuggingFace model ID or path for the base (unquantized) model",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="HuggingFace model ID or path for the target (quantized) model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="HuggingFace dataset ID (default: wikitext)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration (default: wikitext-2-raw-v1)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the text column in the dataset (default: text)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of samples to evaluate (default: 128)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (optional)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    result = evaluate_kl_divergence(
        base_model=args.base_model,
        target_model=args.target_model,
        dataset_id=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        device=args.device,
    )

    print("\n" + result.summary())

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
