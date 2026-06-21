"""Analyze singular value spectrum of activations after layer norms.

Captures activations at:
1. After input layer norm (before self-attention)
2. After post-attention layer norm (before MLP)

For each location, plots:
- Original basis singular value spectrum
- Sequence-length wavelet transform singular value spectrum
"""

import torch
import numpy as np
import pywt
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "/home/brian-dellabetta/projects/llm-compressor")
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

# Config
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "mit-han-lab/pile-val-backup"
NUM_CALIBRATION_SAMPLES = 64
MAX_SEQUENCE_LENGTH = 4096
LAYER_IDX = 15  # Analyze layer 15
WAVELET = "db2"


def collect_layernorm_activations(model, layer_idx, dataloader, device="cuda"):
    """Collect activations after both layer norms in a transformer layer.

    Returns:
        input_norm_acts: activations after input_layernorm (before attention) [bfloat16]
        post_attn_norm_acts: activations after post_attention_layernorm (before MLP) [bfloat16]
    """
    input_norm_acts = []
    post_attn_norm_acts = []

    layer = model.model.layers[layer_idx]

    def input_norm_hook(module, input, output):
        # Store in bfloat16 (original precision) to save memory, promote to float64 only when needed
        act = output.detach().cpu().to(torch.bfloat16)
        if len(act.shape) == 3:
            input_norm_acts.append(act)

    def post_attn_norm_hook(module, input, output):
        # Store in bfloat16 (original precision) to save memory, promote to float64 only when needed
        act = output.detach().cpu().to(torch.bfloat16)
        if len(act.shape) == 3:
            post_attn_norm_acts.append(act)

    # Register hooks on the layer norms
    h1 = layer.input_layernorm.register_forward_hook(input_norm_hook)
    h2 = layer.post_attention_layernorm.register_forward_hook(post_attn_norm_hook)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            model(batch["input_ids"].to(device))

    h1.remove()
    h2.remove()

    return torch.cat(input_norm_acts, dim=0), torch.cat(post_attn_norm_acts, dim=0)


def apply_wavelet_along_sequence(X_3d, wavelet="db2"):
    """Apply 1D wavelet transform along sequence dimension only.

    Args:
        X_3d: (n_samples, seq_len, hidden_dim) [bfloat16]
        wavelet: wavelet type

    Returns:
        transformed: (n_samples, seq_len, hidden_dim) [bfloat16] - same shape!
    """
    n_samples, seq_len, hidden_dim = X_3d.shape

    # Use single-level DWT to preserve exact length
    # For length N: cA (approx) has N/2 coeffs, cD (detail) has N/2 coeffs -> total N
    transformed_samples = []

    for i in range(n_samples):
        sample = X_3d[i]  # (seq_len, hidden_dim)
        transformed_features = []

        for j in range(hidden_dim):
            # Promote to float64 for wavelet transform to avoid precision issues
            sequence = sample[:, j].to(torch.float64).numpy(force=True)
            # Single-level DWT with periodic mode: preserves length exactly
            cA, cD = pywt.dwt(sequence, wavelet, mode="periodic")
            # Concatenate approximation and detail coefficients
            flat_coeffs = np.concatenate([cA, cD])
            transformed_features.append(flat_coeffs)

        # Stack features back
        transformed_sample = np.stack(
            transformed_features, axis=1, dtype=np.float64
        )  # (seq_len, hidden_dim)
        transformed_samples.append(transformed_sample)

    # All samples now have identical length (seq_len)
    # Store result in bfloat16 to save memory (original precision)
    return torch.tensor(np.stack(transformed_samples, axis=0), dtype=torch.bfloat16)


def compute_singular_values_per_sample(X_3d, is_wavelet=False):
    """Compute singular values for each sample separately.

    Args:
        X_3d: (n_samples, seq_len, hidden_dim)
        is_wavelet: if True, wavelet was applied along seq_len, so we analyze the transpose

    Returns:
        sv_mean: mean singular values across samples (shape: min_rank)
        sv_std: std dev of singular values across samples (shape: min_rank)
        sv_min: min singular values across samples (shape: min_rank)
        sv_max: max singular values across samples (shape: min_rank)
    """
    n_samples = X_3d.shape[0]
    all_sv = []

    for i in range(n_samples):
        sample = X_3d[i]  # (seq_len, hidden_dim)

        if is_wavelet:
            # Wavelet was applied along sequence dim, transpose to analyze it
            X_2d = sample.T.to(torch.float64)  # (hidden_dim, seq_len)
        else:
            # Original basis: (seq_len, hidden_dim)
            X_2d = sample.to(torch.float64)

        # Compute SVD in float64 for numerical precision
        _, S, _ = torch.linalg.svd(X_2d, full_matrices=False)
        all_sv.append(S.numpy(force=True).astype(np.float64))

    # Stack and compute statistics across samples
    all_sv = np.array(all_sv)  # (n_samples, rank)
    sv_mean = all_sv.mean(axis=0)
    sv_std = all_sv.std(axis=0)
    sv_min = all_sv.min(axis=0)
    sv_max = all_sv.max(axis=0)

    return sv_mean, sv_std, sv_min, sv_max


def plot_spectrum(ax, sv_mean, sv_std, sv_min, sv_max, title, color="blue"):
    """Plot singular value spectrum on log-log scale with error bands.

    Args:
        ax: matplotlib axis
        sv_mean: mean singular values across samples (float64)
        sv_std: std dev of singular values across samples (float64)
        sv_min: min singular values across samples (float64)
        sv_max: max singular values across samples (float64)
        title: plot title
        color: line color
    """
    # Ensure float64 precision
    sv_mean = sv_mean.astype(np.float64)
    sv_std = sv_std.astype(np.float64)
    sv_min = sv_min.astype(np.float64)
    sv_max = sv_max.astype(np.float64)

    # Normalize by largest singular value (mean)
    normalized_mean = sv_mean / sv_mean[0]
    normalized_std = sv_std / sv_mean[0]
    normalized_min = sv_min / sv_mean[0]
    normalized_max = sv_max / sv_mean[0]

    indices = np.arange(1, len(normalized_mean) + 1, dtype=np.float64)

    # Plot min/max band (outermost)
    ax.fill_between(
        indices,
        np.maximum(normalized_min, 1e-10),
        normalized_max,
        color=color,
        alpha=0.1,
        label="min/max",
    )

    # Plot ±2 std band (middle)
    ax.fill_between(
        indices,
        np.maximum(normalized_mean - 2 * normalized_std, 1e-10),
        normalized_mean + 2 * normalized_std,
        color=color,
        alpha=0.15,
        label="±2 std",
    )

    # Plot ±1 std band (innermost)
    ax.fill_between(
        indices,
        np.maximum(normalized_mean - normalized_std, 1e-10),
        normalized_mean + normalized_std,
        color=color,
        alpha=0.25,
        label="±1 std",
    )

    # Plot mean line on top
    ax.loglog(
        indices, normalized_mean, "-", color=color, linewidth=2, alpha=0.9, label="Mean"
    )

    ax.set_xlabel("Singular Value Index", fontsize=10)
    ax.set_ylabel("Normalized Singular Value", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")

    # Fit to the FULL spectrum - no truncation
    # Remove only zeros or very small values that would cause log issues
    valid_mask = normalized_mean > 1e-10
    tail_indices = indices[valid_mask]
    tail_sv = normalized_mean[valid_mask]

    if len(tail_indices) > 10:
        log_idx = np.log(tail_indices)
        log_sv = np.log(tail_sv)

        # Power law fit in log-log space: log(y) = alpha * log(x) + b
        coeffs_power = np.polyfit(log_idx, log_sv, 1)
        alpha = coeffs_power[0]
        C_power = np.exp(coeffs_power[1])

        # Exponential fit in semi-log space: log(y) = -beta * x + b
        coeffs_exp = np.polyfit(tail_indices, log_sv, 1)
        beta = -coeffs_exp[0]
        C_exp = np.exp(coeffs_exp[1])

        # Compute R² in LOG-LOG space for fair comparison
        power_pred_log = coeffs_power[0] * log_idx + coeffs_power[1]
        exp_pred_log = coeffs_exp[0] * tail_indices + coeffs_exp[1]

        ss_res_power = np.sum((log_sv - power_pred_log) ** 2)
        ss_res_exp = np.sum((log_sv - exp_pred_log) ** 2)
        ss_tot = np.sum((log_sv - log_sv.mean()) ** 2)

        r2_power = 1 - ss_res_power / ss_tot
        r2_exp = 1 - ss_res_exp / ss_tot

        # Plot both fits over the full range
        power_fit = C_power * indices**alpha
        exp_fit = C_exp * np.exp(-beta * indices)

        ax.loglog(
            indices,
            power_fit,
            "--",
            color="red",
            linewidth=1.5,
            alpha=0.7,
            label=f"Power law: α={alpha:.2f} (R²={r2_power:.4f})",
        )
        ax.loglog(
            indices,
            exp_fit,
            ":",
            color="orange",
            linewidth=1.5,
            alpha=0.7,
            label=f"Exponential: β={beta:.2e} (R²={r2_exp:.4f})",
        )

    ax.legend(fontsize=7, loc="best")


def main():
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map="cuda",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading calibration dataset (fixed length = {MAX_SEQUENCE_LENGTH})...")
    ds = load_dataset(DATASET_ID, split=f"validation[:{NUM_CALIBRATION_SAMPLES * 20}]")

    def tokenize_and_filter(example):
        ids = tokenizer.encode(example["text"].strip())
        if len(ids) >= MAX_SEQUENCE_LENGTH:
            return {"input_ids": ids[:MAX_SEQUENCE_LENGTH], "keep": True}
        return {"input_ids": ids, "keep": False}

    ds = ds.map(tokenize_and_filter, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["keep"]).remove_columns(["keep"])
    ds = ds.shuffle(seed=42).select(range(min(NUM_CALIBRATION_SAMPLES, len(ds))))
    print(f"  {len(ds)} samples, all exactly {MAX_SEQUENCE_LENGTH} tokens")

    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
        return {"input_ids": input_ids}

    dataloader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)

    # Collect activations
    print(f"\nCollecting activations from layer {LAYER_IDX} layer norms...")
    input_norm_acts, post_attn_norm_acts = collect_layernorm_activations(
        model, LAYER_IDX, dataloader
    )

    print(f"Input LayerNorm activations: {input_norm_acts.shape}")
    print(f"Post-Attention LayerNorm activations: {post_attn_norm_acts.shape}")

    # Compute singular values for original basis (per-sample)
    print("\nComputing singular values in original basis (per-sample)...")
    input_sv_orig_mean, input_sv_orig_std, input_sv_orig_min, input_sv_orig_max = compute_singular_values_per_sample(
        input_norm_acts
    )
    post_attn_sv_orig_mean, post_attn_sv_orig_std, post_attn_sv_orig_min, post_attn_sv_orig_max = compute_singular_values_per_sample(
        post_attn_norm_acts
    )

    # Apply wavelet transform along sequence dimension
    print(f"\nApplying {WAVELET} wavelet transform along sequence dimension...")
    input_wav = apply_wavelet_along_sequence(input_norm_acts, WAVELET)
    post_attn_wav = apply_wavelet_along_sequence(post_attn_norm_acts, WAVELET)

    print(f"Input LayerNorm (wavelet): {input_wav.shape}")
    print(f"Post-Attention LayerNorm (wavelet): {post_attn_wav.shape}")

    # Compute singular values for wavelet basis (per-sample)
    print("\nComputing singular values in wavelet basis (per-sample)...")
    input_sv_wav_mean, input_sv_wav_std, input_sv_wav_min, input_sv_wav_max = compute_singular_values_per_sample(
        input_wav, is_wavelet=True
    )
    post_attn_sv_wav_mean, post_attn_sv_wav_std, post_attn_sv_wav_min, post_attn_sv_wav_max = compute_singular_values_per_sample(
        post_attn_wav, is_wavelet=True
    )

    # Create plots
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Singular Value Spectrum Analysis - Layer {LAYER_IDX}\n"
        + f"({NUM_CALIBRATION_SAMPLES} samples, seq_len={MAX_SEQUENCE_LENGTH})",
        fontsize=14,
        fontweight="bold",
    )

    # Input LayerNorm - Original
    plot_spectrum(
        axes[0, 0],
        input_sv_orig_mean,
        input_sv_orig_std,
        input_sv_orig_min,
        input_sv_orig_max,
        "Input LayerNorm - Original Basis",
        color="blue",
    )

    # Input LayerNorm - Wavelet
    plot_spectrum(
        axes[0, 1],
        input_sv_wav_mean,
        input_sv_wav_std,
        input_sv_wav_min,
        input_sv_wav_max,
        f"Input LayerNorm - {WAVELET.upper()} Wavelet (seq dim)",
        color="green",
    )

    # Post-Attention LayerNorm - Original
    plot_spectrum(
        axes[1, 0],
        post_attn_sv_orig_mean,
        post_attn_sv_orig_std,
        post_attn_sv_orig_min,
        post_attn_sv_orig_max,
        "Post-Attention LayerNorm - Original Basis",
        color="blue",
    )

    # Post-Attention LayerNorm - Wavelet
    plot_spectrum(
        axes[1, 1],
        post_attn_sv_wav_mean,
        post_attn_sv_wav_std,
        post_attn_sv_wav_min,
        post_attn_sv_wav_max,
        f"Post-Attention LayerNorm - {WAVELET.upper()} Wavelet (seq dim)",
        color="green",
    )

    plt.tight_layout()

    # Save figure
    output_path = "/home/brian-dellabetta/projects/llm-compressor/spectrum_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Print statistics
    print("\n" + "=" * 80)
    print("SINGULAR VALUE STATISTICS - PER-SAMPLE ANALYSIS")
    print("=" * 80)

    for name, sv_mean, sv_std, sv_min, sv_max in [
        ("Input LayerNorm (Original)", input_sv_orig_mean, input_sv_orig_std, input_sv_orig_min, input_sv_orig_max),
        ("Input LayerNorm (Wavelet)", input_sv_wav_mean, input_sv_wav_std, input_sv_wav_min, input_sv_wav_max),
        (
            "Post-Attn LayerNorm (Original)",
            post_attn_sv_orig_mean,
            post_attn_sv_orig_std,
            post_attn_sv_orig_min,
            post_attn_sv_orig_max,
        ),
        ("Post-Attn LayerNorm (Wavelet)", post_attn_sv_wav_mean, post_attn_sv_wav_std, post_attn_sv_wav_min, post_attn_sv_wav_max),
    ]:
        normalized = sv_mean / sv_mean[0]
        normalized_std = sv_std / sv_mean[0]

        # Compute effective rank (participation ratio) from mean
        eff_rank = (sv_mean.sum() ** 2) / (sv_mean**2).sum()

        # Fit power law and exponential to FULL spectrum
        indices = np.arange(1, len(normalized) + 1, dtype=np.float64)

        # Remove only zeros or very small values
        valid_mask = normalized > 1e-10
        tail_indices = indices[valid_mask]
        tail_sv = normalized[valid_mask]

        alpha = beta = r2_power = r2_exp = np.nan
        if len(tail_indices) > 10:
            log_idx = np.log(tail_indices)
            log_sv = np.log(tail_sv)

            # Power law fit in log-log space
            coeffs_power = np.polyfit(log_idx, log_sv, 1)
            alpha = coeffs_power[0]
            power_pred_log = coeffs_power[0] * log_idx + coeffs_power[1]

            # Exponential fit in semi-log space
            coeffs_exp = np.polyfit(tail_indices, log_sv, 1)
            beta = -coeffs_exp[0]
            exp_pred_log = coeffs_exp[0] * tail_indices + coeffs_exp[1]

            # Compare R² in LOG-LOG space (appropriate for log-log visualization)
            ss_tot = np.sum((log_sv - log_sv.mean()) ** 2)
            r2_power = 1 - np.sum((log_sv - power_pred_log) ** 2) / ss_tot
            r2_exp = 1 - np.sum((log_sv - exp_pred_log) ** 2) / ss_tot

        print(f"\n{name}:")
        print(f"  Total singular values: {len(sv_mean)}")
        print(
            f"  Effective rank (mean): {eff_rank:.1f} ({100*eff_rank/len(sv_mean):.1f}%)"
        )
        print(f"\n  Full spectrum decay analysis ({len(tail_indices)} valid indices):")
        print(f"    Power law:     α = {alpha:.3f}  (R² = {r2_power:.4f})")
        print(f"    Exponential:   β = {beta:.2e}  (R² = {r2_exp:.4f})")
        if r2_power > r2_exp:
            print(f"    → Power law fit is better (ΔR² = {r2_power - r2_exp:.4f})")
        else:
            print(f"    → Exponential fit is better (ΔR² = {r2_exp - r2_power:.4f})")
        print(f"\n  Percentiles (mean ± std):")
        print(
            f"    10th:   {normalized[len(sv_mean)//10]:.2e} ± {normalized_std[len(sv_mean)//10]:.2e}"
        )
        print(
            f"    50th:   {np.median(normalized):.2e} ± {normalized_std[len(sv_mean)//2]:.2e}"
        )
        print(
            f"    90th:   {normalized[9*len(sv_mean)//10]:.2e} ± {normalized_std[9*len(sv_mean)//10]:.2e}"
        )
        print(
            f"    99th:   {normalized[99*len(sv_mean)//100]:.2e} ± {normalized_std[99*len(sv_mean)//100]:.2e}"
        )

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
