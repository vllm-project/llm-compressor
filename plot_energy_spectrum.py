"""Adaptive outlier extraction: projection-ratio-based column partitioning.

For each column, computes how well it's captured by the ASVD low-rank subspace.
Columns with low projection ratio are true outliers → extract to sparse.
Columns with high projection ratio are well-aligned → leave for low-rank.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Config — match oneshot_greedy_multiscale.py
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "mit-han-lab/pile-val-backup"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048

COMPRESS_TARGETS = [
    "re:.*model.layers.15.mlp.(gate|up|down)_proj$",
    "re:.*model.layers.15.self_attn.(q|k|v|o)_proj$",
]

import re


def get_calib_dataset(tokenizer):
    ds = load_dataset(DATASET_ID, split=f"validation[:{NUM_CALIBRATION_SAMPLES*10}]")

    def preprocess(example):
        return {
            "input_ids": tokenizer.encode(example["text"].strip()[:MAX_SEQUENCE_LENGTH])
        }

    ds = (
        ds.shuffle(seed=42)
        .map(preprocess, remove_columns=ds.column_names)
        .select(range(NUM_CALIBRATION_SAMPLES))
    )
    return ds


def collect_layer_activations(model, layer_name, dataloader, device="cuda"):
    activations = []
    parts = layer_name.split(".")
    layer = model
    for part in parts:
        layer = layer[int(part)] if part.isdigit() else getattr(layer, part)

    def hook(module, input, output):
        act = input[0].detach().cpu()
        if len(act.shape) == 3:
            act = act.reshape(-1, act.shape[-1])
        activations.append(act)

    handle = layer.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            model(batch["input_ids"].to(device))
    handle.remove()
    return torch.cat(activations, dim=0)


def find_target_layers(model, targets):
    layers = []

    def search(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            matched = False
            for pat in targets:
                if pat.startswith("re:"):
                    if re.match(pat[3:], full_name):
                        matched = True
                elif pat in full_name:
                    matched = True
            if matched and isinstance(child, nn.Linear):
                layers.append((full_name, child))
            else:
                search(child, full_name)

    search(model)
    return layers


def main():
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="cuda",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading calibration dataset...")
    calib_dataset = get_calib_dataset(tokenizer)

    def collate_fn(batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id or 0
        )
        return {"input_ids": input_ids}

    dataloader = DataLoader(calib_dataset, batch_size=4, collate_fn=collate_fn)

    target_layers = find_target_layers(model, COMPRESS_TARGETS)
    print(f"Found {len(target_layers)} layers: {[n for n, _ in target_layers]}")

    # Reference rank for projection-ratio computation (fraction of min dimension)
    REFERENCE_RANK_FRAC = 0.30

    def safe_svd(matrix):
        """SVD with fallbacks for ill-conditioned matrices."""
        try:
            return torch.linalg.svd(matrix, full_matrices=False)
        except torch._C._LinAlgError:
            pass
        # Fallback 1: try on CUDA with gesvd driver
        if torch.cuda.is_available():
            try:
                U, S, Vh = torch.linalg.svd(
                    matrix.cuda(), full_matrices=False, driver="gesvd"
                )
                return U.cpu(), S.cpu(), Vh.cpu()
            except Exception:
                pass
        # Fallback 2: numpy SVD
        U_np, S_np, Vh_np = np.linalg.svd(matrix.numpy(), full_matrices=False)
        return (
            torch.from_numpy(U_np),
            torch.from_numpy(S_np),
            torch.from_numpy(Vh_np),
        )

    def effective_rank(singular_values):
        """Effective rank via Shannon entropy of normalized singular value energy.

        p_i = sigma_i^2 / sum(sigma_j^2)
        H = -sum(p_i * ln(p_i))
        ER = exp(H)
        """
        sv2 = singular_values.float() ** 2
        sv2 = sv2[sv2 > 0]  # drop zeros
        p = sv2 / sv2.sum()
        H = -(p * p.log()).sum().item()
        H_max = np.log(len(p))
        H_norm = H / H_max if H_max > 0 else 0.0
        ER = np.exp(H)
        return ER, H_norm

    fig, axes = plt.subplots(
        3, len(target_layers), figsize=(6 * len(target_layers), 15)
    )
    if len(target_layers) == 1:
        axes = axes.reshape(3, 1)

    for col, (layer_name, layer) in enumerate(target_layers):
        print(f"\n{'='*70}")
        print(f"Layer: {layer_name} ({layer.weight.shape})")
        print(f"{'='*70}")

        # Collect activations
        input_acts = collect_layer_activations(model, layer_name, dataloader)
        print(f"  Collected {input_acts.shape[0]} activation samples")

        weight = layer.weight.detach().float().cpu()
        input_acts_cpu = input_acts.float().cpu()
        out_features, in_features = weight.shape
        original_params = weight.numel()
        min_dim = min(out_features, in_features)

        # Activation norms for importance weighting
        act_norms = input_acts_cpu.norm(dim=0)
        act_norms_safe = act_norms.clamp(min=1e-10)

        # ==========================================
        # Baseline: SVD and ASVD of original weight
        # ==========================================
        U_o, S_o, Vh_o = safe_svd(weight)
        sv_o_energy = S_o**2
        sv_o_cumulative = torch.cumsum(sv_o_energy, dim=0) / sv_o_energy.sum()

        W_scaled_orig = weight * act_norms_safe.unsqueeze(0)
        U_ao, S_ao, Vh_ao = safe_svd(W_scaled_orig)
        sv_ao_energy = S_ao**2
        sv_ao_cumulative = torch.cumsum(sv_ao_energy, dim=0) / sv_ao_energy.sum()

        er_svd, hn_svd = effective_rank(S_o)
        er_asvd, hn_asvd = effective_rank(S_ao)

        print(f"\n  Baseline spectrum:")
        print(f"    Effective rank: SVD={er_svd:.0f}/{min_dim} (H_norm={hn_svd:.3f}), "
              f"ASVD={er_asvd:.0f}/{min_dim} (H_norm={hn_asvd:.3f})")
        for thresh in [0.90, 0.95, 0.99]:
            n_svd = int((sv_o_cumulative < thresh).sum().item()) + 1
            n_asvd = int((sv_ao_cumulative < thresh).sum().item()) + 1
            print(f"    {thresh:.0%} energy: SVD={n_svd}, ASVD={n_asvd} modes")

        # ==========================================
        # Projection-ratio analysis
        # ==========================================
        ref_rank = max(1, int(REFERENCE_RANK_FRAC * min_dim))
        print(f"\n  Projection-ratio analysis (reference rank k={ref_rank}, {REFERENCE_RANK_FRAC:.0%} of {min_dim}):")

        # ASVD left singular vectors at reference rank
        U_k = U_ao[:, :ref_rank]  # (out_features, ref_rank)

        # For each column w_j, compute projection ratio:
        #   proj_ratio_j = ||U_k^T @ w_j||^2 / ||w_j||^2
        # This measures how much of column j lies in the ASVD low-rank subspace.
        col_norms_sq = (weight ** 2).sum(dim=0)  # (in_features,)
        col_norms_sq_safe = col_norms_sq.clamp(min=1e-20)

        # U_k^T @ W = (ref_rank, out_features) @ (out_features, in_features) = (ref_rank, in_features)
        proj = U_k.T @ weight  # (ref_rank, in_features)
        proj_norms_sq = (proj ** 2).sum(dim=0)  # (in_features,)

        proj_ratios = proj_norms_sq / col_norms_sq_safe  # (in_features,)
        proj_ratios = proj_ratios.clamp(0, 1)

        # Statistics
        print(f"    Projection ratio stats:")
        print(f"      min={proj_ratios.min():.4f}, max={proj_ratios.max():.4f}, "
              f"mean={proj_ratios.mean():.4f}, median={proj_ratios.median():.4f}")

        # Distribution buckets
        for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            n_below = (proj_ratios < thresh).sum().item()
            print(f"      ratio < {thresh}: {n_below} cols ({n_below/in_features:.1%})")

        # ==========================================
        # Adaptive extraction: extract columns with low projection ratio
        # Compare against importance-based extraction (current approach)
        # ==========================================
        print(f"\n  Adaptive vs importance-based extraction:")

        # Importance-based: activation-weighted column norm
        col_importance = weight.norm(dim=0) * act_norms_safe

        # Test multiple thresholds
        PROJ_THRESHOLDS = [0.1, 0.3, 0.5, 0.7]
        # Also test equivalent importance-based fractions for comparison

        adaptive_results = {}
        importance_results = {}

        for proj_thresh in PROJ_THRESHOLDS:
            # Adaptive: extract columns with projection ratio < threshold
            outlier_mask = proj_ratios < proj_thresh
            num_outliers = outlier_mask.sum().item()
            if num_outliers == 0:
                continue

            sparse_frac = num_outliers / in_features

            # --- Adaptive extraction ---
            W_sparse_adapt = torch.zeros_like(weight)
            W_sparse_adapt[:, outlier_mask] = weight[:, outlier_mask]
            sparse_params = int(num_outliers) * out_features

            W_cleaned_adapt = weight.clone()
            W_cleaned_adapt[:, outlier_mask] = 0

            # ASVD of adaptive-cleaned matrix
            W_ca = W_cleaned_adapt * act_norms_safe.unsqueeze(0)
            U_ca, S_ca, Vh_ca = safe_svd(W_ca)
            sv_ca_energy = S_ca**2
            sv_ca_cumulative = torch.cumsum(sv_ca_energy, dim=0) / sv_ca_energy.sum()
            er_ca, hn_ca = effective_rank(S_ca)

            # SVD of adaptive-cleaned matrix
            U_c, S_c, Vh_c = safe_svd(W_cleaned_adapt)
            sv_c_energy = S_c**2
            sv_c_cumulative = torch.cumsum(sv_c_energy, dim=0) / sv_c_energy.sum()
            er_c, hn_c = effective_rank(S_c)

            sparse_snr = 20 * torch.log10(
                weight.norm() / (W_cleaned_adapt.norm() + 1e-10)
            ).item()

            adaptive_results[proj_thresh] = {
                "svd_cumulative": sv_c_cumulative,
                "asvd_cumulative": sv_ca_cumulative,
                "num_cols": int(num_outliers),
                "sparse_frac": sparse_frac,
                "sparse_params": sparse_params,
                "er_svd": er_c,
                "er_asvd": er_ca,
                "hn_svd": hn_c,
                "hn_asvd": hn_ca,
                "sparse_snr": sparse_snr,
            }

            # --- Importance-based extraction with same number of columns ---
            top_cols = torch.argsort(col_importance, descending=True)[:int(num_outliers)]
            W_cleaned_imp = weight.clone()
            W_cleaned_imp[:, top_cols] = 0

            W_ci = W_cleaned_imp * act_norms_safe.unsqueeze(0)
            U_ci, S_ci, Vh_ci = safe_svd(W_ci)
            sv_ci_energy = S_ci**2
            sv_ci_cumulative = torch.cumsum(sv_ci_energy, dim=0) / sv_ci_energy.sum()
            er_ci, hn_ci = effective_rank(S_ci)

            _, S_ci_svd, _ = safe_svd(W_cleaned_imp)
            sv_ci_svd_energy = S_ci_svd**2
            sv_ci_svd_cumulative = torch.cumsum(sv_ci_svd_energy, dim=0) / sv_ci_svd_energy.sum()
            er_ci_svd, hn_ci_svd = effective_rank(S_ci_svd)

            imp_sparse_snr = 20 * torch.log10(
                weight.norm() / (W_cleaned_imp.norm() + 1e-10)
            ).item()

            importance_results[proj_thresh] = {
                "svd_cumulative": sv_ci_svd_cumulative,
                "asvd_cumulative": sv_ci_cumulative,
                "er_svd": er_ci_svd,
                "er_asvd": er_ci,
                "hn_svd": hn_ci_svd,
                "hn_asvd": hn_ci,
                "sparse_snr": imp_sparse_snr,
            }

            # Overlap: how many columns are in common?
            imp_set = set(top_cols.tolist())
            adapt_set = set(torch.where(outlier_mask)[0].tolist())
            overlap = len(imp_set & adapt_set)

            print(f"\n    proj_ratio < {proj_thresh}: {num_outliers} cols ({sparse_frac:.1%}), "
                  f"{sparse_params:,} params ({sparse_params/original_params:.1%})")
            print(f"      Adaptive:   sparse SNR={sparse_snr:.1f}dB, "
                  f"ER: SVD {er_svd:.0f}→{er_c:.0f} (H={hn_c:.3f}), "
                  f"ASVD {er_asvd:.0f}→{er_ca:.0f} (H={hn_ca:.3f})")
            print(f"      Importance: sparse SNR={imp_sparse_snr:.1f}dB, "
                  f"ER: SVD {er_svd:.0f}→{er_ci_svd:.0f} (H={hn_ci_svd:.3f}), "
                  f"ASVD {er_asvd:.0f}→{er_ci:.0f} (H={hn_ci:.3f})")
            print(f"      Column overlap: {overlap}/{num_outliers} ({overlap/num_outliers:.0%})")

            for thresh in [0.90, 0.95, 0.99]:
                n_adapt = int((sv_ca_cumulative < thresh).sum().item()) + 1
                n_imp = int((sv_ci_cumulative < thresh).sum().item()) + 1
                n_orig = int((sv_ao_cumulative < thresh).sum().item()) + 1
                print(f"      {thresh:.0%} ASVD energy: orig={n_orig}, adaptive={n_adapt}, importance={n_imp}")

        # ==========================================
        # Find optimal threshold: minimize ASVD effective rank
        # ==========================================
        print(f"\n  Sweep: ASVD ER vs projection-ratio threshold:")
        sweep_thresholds = np.arange(0.05, 1.0, 0.05)
        sweep_ers = []
        sweep_ncols = []
        for t in sweep_thresholds:
            mask = proj_ratios < t
            nc = mask.sum().item()
            if nc == 0 or nc >= in_features:
                sweep_ers.append(er_asvd)
                sweep_ncols.append(nc)
                continue
            W_sw = weight.clone()
            W_sw[:, mask] = 0
            W_sw_scaled = W_sw * act_norms_safe.unsqueeze(0)
            _, S_sw, _ = safe_svd(W_sw_scaled)
            er_sw, _ = effective_rank(S_sw)
            sweep_ers.append(er_sw)
            sweep_ncols.append(nc)
            print(f"    thresh={t:.2f}: {nc} cols ({nc/in_features:.1%}) → ASVD ER={er_sw:.0f}")

        best_idx = int(np.argmin(sweep_ers))
        best_thresh = sweep_thresholds[best_idx]
        best_er = sweep_ers[best_idx]
        best_ncols = sweep_ncols[best_idx]
        print(f"  → Optimal threshold: {best_thresh:.2f} ({best_ncols} cols, {best_ncols/in_features:.1%}) → ASVD ER={best_er:.0f} (from {er_asvd:.0f})")

        # ==========================================
        # Plot Row 0: Projection ratio distribution
        # ==========================================
        ax = axes[0, col]
        ax.hist(proj_ratios.numpy(), bins=100, color="steelblue", alpha=0.7, edgecolor="none")
        for pt in PROJ_THRESHOLDS:
            if pt in adaptive_results:
                ax.axvline(x=pt, color="red", linestyle="--", alpha=0.6,
                           label=f"thresh={pt} ({adaptive_results[pt]['num_cols']} cols)")
        ax.axvline(x=best_thresh, color="green", linewidth=2, linestyle="-",
                   label=f"optimal={best_thresh:.2f} ({best_ncols} cols)")
        ax.set_xlabel("Projection ratio (onto ASVD rank-k subspace)")
        ax.set_ylabel("Number of columns")
        ax.set_title(f"Projection ratio distribution\n{layer_name}\n(k={ref_rank}, {out_features}x{in_features})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ==========================================
        # Plot Row 1: ASVD cumulative energy — adaptive vs importance
        # ==========================================
        ax = axes[1, col]
        x_range = np.arange(1, len(sv_ao_cumulative) + 1)
        ax.plot(x_range, sv_ao_cumulative.numpy(), label="Original ASVD", color="black", linewidth=2, alpha=0.8)

        colors_adapt = plt.cm.Blues(np.linspace(0.4, 0.9, len(adaptive_results)))
        colors_imp = plt.cm.Reds(np.linspace(0.4, 0.9, len(importance_results)))
        for i, pt in enumerate(sorted(adaptive_results.keys())):
            r = adaptive_results[pt]
            ax.plot(x_range, r["asvd_cumulative"].numpy(),
                    label=f"Adaptive <{pt} ({r['num_cols']}col, ER={r['er_asvd']:.0f})",
                    color=colors_adapt[i], alpha=0.7, linewidth=1.5)
        for i, pt in enumerate(sorted(importance_results.keys())):
            r = importance_results[pt]
            ax.plot(x_range, r["asvd_cumulative"].numpy(),
                    label=f"Importance ({adaptive_results[pt]['num_cols']}col, ER={r['er_asvd']:.0f})",
                    color=colors_imp[i], alpha=0.5, linewidth=1, linestyle="--")
        ax.axhline(y=0.90, color="gray", linestyle=":", alpha=0.4)
        ax.axhline(y=0.99, color="gray", linestyle=":", alpha=0.4)
        ax.set_xlabel("Number of singular values")
        ax.set_ylabel("Cumulative energy fraction")
        ax.set_title(f"ASVD: adaptive vs importance extraction\n{layer_name}")
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.3)

        # ==========================================
        # Plot Row 2: ASVD ER sweep vs threshold
        # ==========================================
        ax = axes[2, col]
        ax.plot(sweep_thresholds, sweep_ers, color="steelblue", linewidth=2, marker="o", markersize=3)
        ax.axhline(y=er_asvd, color="black", linestyle="--", alpha=0.5, label=f"Original ER={er_asvd:.0f}")
        ax.axvline(x=best_thresh, color="green", linestyle="-", alpha=0.7,
                   label=f"Best: {best_thresh:.2f} → ER={best_er:.0f}")
        ax.set_xlabel("Projection-ratio threshold")
        ax.set_ylabel("ASVD Effective Rank")
        ax.set_title(f"ASVD ER vs extraction threshold\n{layer_name}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Secondary x-axis: number of columns extracted
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_positions = sweep_thresholds[::4]
        tick_labels = [str(sweep_ncols[i]) for i in range(0, len(sweep_ncols), 4)]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, fontsize=7)
        ax2.set_xlabel("Columns extracted", fontsize=8)

    plt.tight_layout()
    plt.savefig("energy_spectrum.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to energy_spectrum.png")
    plt.close()


if __name__ == "__main__":
    main()
