"""
Compute KL divergence between two models using saved hidden states.

This script loads hidden states extracted by extract_hidden_states.py,
applies the lm_head to reconstruct logits, and computes KL divergence
between the base and target model distributions.

Hidden states are post-norm (extracted at layer num_hidden_layers from vLLM),
so no normalization is needed. Only the lm_head weight is loaded from the
model checkpoint.

Usage:
    python tools/kl_divergence/compute_kl.py \
        --base-dir ./hidden_states/base \
        --target-dir ./hidden_states/quantized \
        --base-model meta-llama/Meta-Llama-3-8B-Instruct \
        --target-model ./Meta-Llama-3-8B-Instruct-W4A16 \
        --temperature 1.0 \
        --device cuda:0 \
        --output results.json
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm

# Support running from repo root or from tools/kl_divergence/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lm_head_utils import load_lm_head_weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute KL divergence from extracted hidden states"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Directory containing base model hidden states",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="Directory containing target model hidden states",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model ID (for loading lm_head weights)",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="Target model ID (default: same as base-model, for shared lm_head)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax (default: 1.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for computation (default: cuda:0 if available)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (default: print to stdout only)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Number of tokens to process at once for logit computation "
        "(lower = less memory, default: 64)",
    )
    # Architecture overrides
    parser.add_argument(
        "--lm-head-weight-name",
        type=str,
        default=None,
        help="Override for lm_head weight tensor name",
    )
    parser.add_argument(
        "--lm-head-bias-name",
        type=str,
        default=None,
        help="Override for lm_head bias tensor name",
    )
    parser.add_argument(
        "--embed-weight-name",
        type=str,
        default=None,
        help="Override for embedding weight tensor name (for tied embeddings)",
    )
    return parser.parse_args()


def _compute_kl_for_chunk(
    base_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    base_weights: dict,
    target_weights: dict,
    temperature: float,
) -> torch.Tensor:
    """
    Compute per-position KL divergence for a chunk of hidden states.

    Hidden states are expected to be post-norm (extracted at layer index
    num_hidden_layers from vLLM), so no additional normalization is applied.

    :param base_hidden: [chunk_size, hidden_dim] base hidden states (post-norm)
    :param target_hidden: [chunk_size, hidden_dim] target hidden states (post-norm)
    :param base_weights: dict with lm_head_weight, lm_head_bias
    :param target_weights: dict with lm_head_weight, lm_head_bias
    :param temperature: temperature for softmax scaling
    :return: [chunk_size] per-position KL divergence values
    """
    # Compute logits: [chunk_size, vocab_size]
    # Hidden states are already post-norm, so apply lm_head directly
    # Weights are pre-cast to float32 in compute_kl_divergence
    base_logits = base_hidden.float() @ base_weights["lm_head_weight"].T
    if base_weights["lm_head_bias"] is not None:
        base_logits += base_weights["lm_head_bias"]

    target_logits = target_hidden.float() @ target_weights["lm_head_weight"].T
    if target_weights["lm_head_bias"] is not None:
        target_logits += target_weights["lm_head_bias"]

    # Apply temperature
    if temperature != 1.0:
        base_logits = base_logits / temperature
        target_logits = target_logits / temperature

    # Compute KL(P_base || Q_target) per position
    # = sum_x P_base(x) * (log P_base(x) - log Q_target(x))
    base_log_probs = F.log_softmax(base_logits, dim=-1)
    target_log_probs = F.log_softmax(target_logits, dim=-1)

    # Using log-space computation for numerical stability
    kl_per_position = F.kl_div(
        target_log_probs, base_log_probs, reduction="none", log_target=True
    ).sum(dim=-1)

    return kl_per_position


def compute_kl_divergence(
    base_dir: str,
    target_dir: str,
    base_model: str,
    target_model: str = None,
    temperature: float = 1.0,
    device: str = "cuda:0",
    chunk_size: int = 64,
    lm_head_weight_name: str = None,
    lm_head_bias_name: str = None,
    embed_weight_name: str = None,
) -> dict:
    """
    Compute KL divergence between two models using their saved hidden states.

    :param base_dir: directory with base model hidden states
    :param target_dir: directory with target model hidden states
    :param base_model: base model ID (for lm_head weights)
    :param target_model: target model ID (default: same as base_model)
    :param temperature: softmax temperature
    :param device: computation device
    :param chunk_size: tokens per chunk for logit computation
    :param lm_head_weight_name: override for lm_head weight tensor name
    :param lm_head_bias_name: override for lm_head bias tensor name
    :param embed_weight_name: override for embed weight tensor name
    :return: dict with mean_kl, std_kl, median_kl, per_sample_kl, metadata
    """
    if target_model is None:
        target_model = base_model

    # Load metadata
    base_meta_path = os.path.join(base_dir, "metadata.json")
    target_meta_path = os.path.join(target_dir, "metadata.json")

    with open(base_meta_path) as f:
        base_meta = json.load(f)
    with open(target_meta_path) as f:
        target_meta = json.load(f)

    # Validate compatibility
    if base_meta["num_samples"] != target_meta["num_samples"]:
        raise ValueError(
            f"Sample count mismatch: base has {base_meta['num_samples']}, "
            f"target has {target_meta['num_samples']}"
        )

    if base_meta["dataset_name"] != target_meta["dataset_name"]:
        print(
            f"Warning: Datasets differ - base: {base_meta['dataset_name']}, "
            f"target: {target_meta['dataset_name']}"
        )

    # Load lm_head weights
    weight_kwargs = dict(
        lm_head_weight_name=lm_head_weight_name,
        lm_head_bias_name=lm_head_bias_name,
        embed_weight_name=embed_weight_name,
    )

    print(f"Loading lm_head weights from: {base_model}")
    base_weights = load_lm_head_weights(base_model, device=device, **weight_kwargs)
    # Pre-cast to float32 once to avoid repeated casting per chunk
    base_weights["lm_head_weight"] = base_weights["lm_head_weight"].float()
    if base_weights["lm_head_bias"] is not None:
        base_weights["lm_head_bias"] = base_weights["lm_head_bias"].float()

    if target_model == base_model:
        target_weights = base_weights
        print("Using shared lm_head weights (same model)")
    else:
        print(f"Loading lm_head weights from: {target_model}")
        target_weights = load_lm_head_weights(
            target_model, device=device, **weight_kwargs
        )
        target_weights["lm_head_weight"] = target_weights["lm_head_weight"].float()
        if target_weights["lm_head_bias"] is not None:
            target_weights["lm_head_bias"] = target_weights["lm_head_bias"].float()

    # Validate vocab size compatibility
    base_vocab = base_weights["lm_head_weight"].shape[0]
    target_vocab = target_weights["lm_head_weight"].shape[0]
    if base_vocab != target_vocab:
        raise ValueError(
            f"Vocab size mismatch: base has {base_vocab}, target has {target_vocab}. "
            "KL divergence requires identical vocabulary."
        )

    # Process hidden state files
    base_files = sorted(base_meta["files"])
    target_files = sorted(target_meta["files"])

    per_sample_kl = []
    total_kl_sum = 0.0
    total_tokens = 0
    start_time = time.time()

    for base_file, target_file in tqdm(
        zip(base_files, target_files),
        total=len(base_files),
        desc="Computing KL divergence",
    ):
        base_path = os.path.join(base_dir, base_file)
        target_path = os.path.join(target_dir, target_file)

        # Load hidden states and token IDs
        with safe_open(base_path, framework="pt") as f:
            base_hidden = f.get_tensor("hidden_states")  # [seq_len, 1, hidden_dim]
            base_token_ids = (
                f.get_tensor("token_ids") if "token_ids" in f.keys() else None
            )
        with safe_open(target_path, framework="pt") as f:
            target_hidden = f.get_tensor("hidden_states")
            target_token_ids = (
                f.get_tensor("token_ids") if "token_ids" in f.keys() else None
            )

        # Validate token alignment
        if base_token_ids is not None and target_token_ids is not None:
            if not torch.equal(base_token_ids, target_token_ids):
                raise ValueError(
                    f"Token ID mismatch between {base_file} and {target_file}. "
                    "Base and target must be extracted from the same input tokens."
                )

        # Squeeze the layer dimension
        base_hidden = base_hidden.squeeze(1)  # [seq_len, hidden_dim]
        target_hidden = target_hidden.squeeze(1)

        seq_len = base_hidden.shape[0]
        if seq_len != target_hidden.shape[0]:
            raise ValueError(
                f"Sequence length mismatch in {base_file}: "
                f"base has {seq_len}, target has {target_hidden.shape[0]}. "
                "Samples must have identical lengths for KL computation."
            )

        # Validate hidden dim matches lm_head
        base_hdim = base_hidden.shape[-1]
        target_hdim = target_hidden.shape[-1]
        if base_hdim != base_weights["lm_head_weight"].shape[1]:
            raise ValueError(
                f"Hidden dim mismatch: base hidden states have dim {base_hdim} "
                f"but lm_head expects {base_weights['lm_head_weight'].shape[1]}"
            )
        if target_hdim != target_weights["lm_head_weight"].shape[1]:
            raise ValueError(
                f"Hidden dim mismatch: target hidden states have dim {target_hdim} "
                f"but lm_head expects {target_weights['lm_head_weight'].shape[1]}"
            )

        # Process in chunks to manage memory
        sample_kl_values = []
        for i in range(0, seq_len, chunk_size):
            chunk_base = base_hidden[i : i + chunk_size].to(device)
            chunk_target = target_hidden[i : i + chunk_size].to(device)

            kl_values = _compute_kl_for_chunk(
                chunk_base,
                chunk_target,
                base_weights,
                target_weights,
                temperature,
            )
            sample_kl_values.append(kl_values)

        # Per-sample and global KL tracking
        all_kl = torch.cat(sample_kl_values)
        sample_mean_kl = all_kl.mean().item()
        per_sample_kl.append(sample_mean_kl)
        total_kl_sum += all_kl.sum().item()
        total_tokens += seq_len

    elapsed = time.time() - start_time

    if not per_sample_kl:
        raise ValueError(
            "No comparable samples were processed. "
            "Check sequence lengths and extracted artifacts."
        )

    # Token-weighted mean KL (more accurate than averaging per-sample means)
    token_weighted_mean_kl = total_kl_sum / total_tokens if total_tokens > 0 else 0.0

    # Aggregate statistics
    kl_tensor = torch.tensor(per_sample_kl)
    results = {
        "mean_kl": token_weighted_mean_kl,
        "mean_kl_per_sample": kl_tensor.mean().item(),
        "std_kl": kl_tensor.std().item() if len(kl_tensor) > 1 else 0.0,
        "median_kl": kl_tensor.median().item(),
        "min_kl": kl_tensor.min().item(),
        "max_kl": kl_tensor.max().item(),
        "num_samples": len(per_sample_kl),
        "total_tokens": total_tokens,
        "temperature": temperature,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0,
        "per_sample_kl": per_sample_kl,
        "base_model": base_meta.get("model_id", base_model),
        "target_model": target_meta.get("model_id", target_model),
        "dataset": base_meta.get("dataset_name", "unknown"),
    }

    return results


def main():
    args = parse_args()

    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    results = compute_kl_divergence(
        base_dir=args.base_dir,
        target_dir=args.target_dir,
        base_model=args.base_model,
        target_model=args.target_model,
        temperature=args.temperature,
        device=args.device,
        chunk_size=args.chunk_size,
        lm_head_weight_name=args.lm_head_weight_name,
        lm_head_bias_name=args.lm_head_bias_name,
        embed_weight_name=args.embed_weight_name,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("KL Divergence Results")
    print("=" * 60)
    print(f"  Base model:    {results['base_model']}")
    print(f"  Target model:  {results['target_model']}")
    print(f"  Dataset:       {results['dataset']}")
    print(f"  Temperature:   {results['temperature']}")
    print(f"  Samples:       {results['num_samples']}")
    print(f"  Total tokens:  {results['total_tokens']}")
    print(
        f"  Time:          {results['elapsed_seconds']:.1f}s "
        f"({results['tokens_per_second']:.0f} tok/s)"
    )
    print("-" * 60)
    print(f"  Mean KL (token-weighted): {results['mean_kl']:.6f}")
    print(f"  Mean KL (per-sample):     {results['mean_kl_per_sample']:.6f}")
    print(f"  Std KL:        {results['std_kl']:.6f}")
    print(f"  Median KL:     {results['median_kl']:.6f}")
    print(f"  Min KL:        {results['min_kl']:.6f}")
    print(f"  Max KL:        {results['max_kl']:.6f}")
    print("=" * 60)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
