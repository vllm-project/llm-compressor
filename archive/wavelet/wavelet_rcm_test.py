"""Test 2D wavelet sparsity per-sample (seq_len × hidden) for X and Y activations,
plus SVD + 1D wavelet hybrid comparison."""

import torch
import numpy as np
import pywt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import sys

sys.path.insert(0, "/home/brian-dellabetta/projects/llm-compressor")
from wavelet_cascade import (
    _effective_rank,
    wavelet_decompose_2d,
    coeffs_to_array_2d,
)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

# Config
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "mit-han-lab/pile-val-backup"
NUM_CALIBRATION_SAMPLES = 64
MAX_SEQUENCE_LENGTH = 2048
LAYER_NAMES = [
    "model.layers.15.self_attn.q_proj",
    "model.layers.15.self_attn.k_proj",
    "model.layers.15.self_attn.v_proj",
    "model.layers.15.self_attn.o_proj",
    "model.layers.15.mlp.gate_proj",
    "model.layers.15.mlp.up_proj",
    "model.layers.15.mlp.down_proj",
]

ENERGY_THRESHOLDS = [0.90, 0.95, 0.99, 0.999, 0.9999]


def fmt_energy(t):
    if t >= 0.9999:
        return "99.99%"
    elif t >= 0.999:
        return "99.9%"
    elif t >= 0.99:
        return "99%"
    elif t >= 0.95:
        return "95%"
    else:
        return f"{t*100:.0f}%"


def compute_rcm_permutation(matrix_2d, threshold_pct=95):
    """Compute RCM column permutation from a 2D (flattened) activation matrix."""
    M = matrix_2d.float().cpu()
    cov = (M.T @ M) / M.shape[0]
    adj = cov.abs().numpy()
    threshold = np.percentile(adj, threshold_pct)
    adj[adj < threshold] = 0
    sparse_adj = csr_matrix(adj)
    perm = reverse_cuthill_mckee(sparse_adj)
    return perm.copy()


def measure_2d_sparsity_per_sample(X_3d, wavelet):
    """Apply 2D wavelet to each sample (seq_len × hidden) and measure sparsity.

    Returns per-threshold: mean_pct, std_pct across samples.
    """
    n_samples = X_3d.shape[0]
    per_sample = {t: [] for t in ENERGY_THRESHOLDS}

    for i in range(n_samples):
        sample = X_3d[i].float().cpu()  # (seq_len, hidden)
        coeffs, _ = wavelet_decompose_2d(sample, wavelet)
        if coeffs is None:
            continue
        flat = coeffs_to_array_2d(coeffs)
        mags = np.abs(flat)
        total_coeffs = len(mags)
        sorted_mags = np.sort(mags)[::-1]
        energy_cumsum = np.cumsum(sorted_mags**2)
        total_energy = energy_cumsum[-1]

        if total_energy < 1e-20:
            for t in ENERGY_THRESHOLDS:
                per_sample[t].append(0.0)
            continue

        for t in ENERGY_THRESHOLDS:
            target = t * total_energy
            num = np.searchsorted(energy_cumsum, target) + 1
            per_sample[t].append(100.0 * num / total_coeffs)

    summary = {}
    for t in ENERGY_THRESHOLDS:
        arr = np.array(per_sample[t])
        summary[t] = {"mean_pct": arr.mean(), "std_pct": arr.std()}
    return summary


def measure_svd_wavelet_hybrid(Y_3d, wavelet):
    """Project Y onto shared SVD basis, then measure 1D wavelet sparsity of temporal signals.

    Picks ranks based on energy thresholds (rank needed for 90%, 95%, 99% SVD energy)
    so the comparison to 2D wavelets is fair. For each rank, measures how much
    1D wavelet compression further reduces the per-sample cost.
    """
    n_samples, seq_len, out_hidden = Y_3d.shape
    orig_params = seq_len * out_hidden

    Y_2d = Y_3d.reshape(-1, out_hidden).float().cpu()
    _, S, Vt = torch.linalg.svd(Y_2d, full_matrices=False)
    er = _effective_rank(S)
    total_energy = (S**2).sum().item()
    cum_energy_ratio = torch.cumsum(S**2, dim=0) / total_energy

    # Pick ranks based on energy thresholds instead of ER multiples
    svd_energy_targets = [0.90, 0.95, 0.99, 0.999]
    ranks = []
    for target in svd_energy_targets:
        r = int((cum_energy_ratio >= target).nonzero(as_tuple=True)[0][0].item()) + 1
        r = min(r, min(out_hidden // 2, 512))
        ranks.append(r)
    ranks = sorted(set(ranks))

    results = {}
    for r in ranks:
        svd_ret = cum_energy_ratio[r - 1].item()
        V_r = Vt[:r, :].T.cpu()  # (out_hidden, r)
        V_cost = r * out_hidden
        svd_only_cost = V_cost + seq_len * r

        per_sample = {t: [] for t in ENERGY_THRESHOLDS}

        for i in range(n_samples):
            projected = Y_3d[i].float().cpu() @ V_r  # (seq_len, r)

            all_coeffs = []
            for col in range(r):
                coeffs = pywt.wavedec(projected[:, col].numpy(), wavelet)
                all_coeffs.extend(coeffs)
            all_flat = np.concatenate(all_coeffs)

            mags = np.abs(all_flat)
            n_coeffs = len(mags)
            sorted_mags = np.sort(mags)[::-1]
            cumsum = np.cumsum(sorted_mags**2)
            proj_energy = cumsum[-1]

            for t in ENERGY_THRESHOLDS:
                required = t / svd_ret
                if required > 1.0 or proj_energy < 1e-20:
                    per_sample[t].append(n_coeffs)
                else:
                    target_e = required * proj_energy
                    k = min(np.searchsorted(cumsum, target_e) + 1, n_coeffs)
                    per_sample[t].append(k)

        summary = {}
        for t in ENERGY_THRESHOLDS:
            counts = np.array(per_sample[t], dtype=float)
            hybrid_cost = V_cost + counts
            hybrid_pct = 100.0 * hybrid_cost / orig_params
            achievable = svd_ret >= t
            summary[t] = {
                "hybrid_pct_mean": hybrid_pct.mean(),
                "hybrid_pct_std": hybrid_pct.std(),
                "achievable": achievable,
            }

        results[r] = {
            "summary": summary,
            "svd_retention": svd_ret * 100,
            "svd_only_pct": 100.0 * svd_only_cost / orig_params,
        }

    return results, er, ranks


def collect_activations(model, layer_name, dataloader, device="cuda"):
    """Collect activations as 3D tensor (n_samples, seq_len, hidden)."""
    activations = []
    parts = layer_name.split(".")
    layer = model
    for part in parts:
        layer = layer[int(part)] if part.isdigit() else getattr(layer, part)

    def hook(module, input, output):
        act = input[0].detach().cpu()
        if len(act.shape) == 3:
            activations.append(act)

    handle = layer.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            model(batch["input_ids"].to(device))
    handle.remove()
    return torch.cat(activations, dim=0)


def print_table(layer_name, X_3d, W, wavelet):
    """Print per-sample 2D wavelet sparsity for X and Y, with and without RCM."""
    n_samples, seq_len, hidden = X_3d.shape

    # Compute Y_3d
    with torch.no_grad():
        Y_3d = (X_3d.float() @ W.float().T).float()
    out_hidden = Y_3d.shape[2]

    print(f"\n{'='*160}")
    print(
        f"Per-Sample 2D Wavelet Sparsity: {layer_name}  ({n_samples} samples, [{seq_len}, {hidden}] → [{seq_len}, {out_hidden}], {wavelet})"
    )
    print(f"{'='*160}")

    # Compute RCM permutations from flattened activations (all samples pooled)
    X_2d = X_3d.reshape(-1, hidden)
    Y_2d = Y_3d.reshape(-1, out_hidden)

    print(f"  Computing RCM for X cols...", end="", flush=True)
    x_perm = compute_rcm_permutation(X_2d)
    print(f" done. Y cols...", end="", flush=True)
    y_perm = compute_rcm_permutation(Y_2d)
    print(f" done.")

    # Measure sparsity for each config
    configs = ["Original", "RCM cols"]

    x_orig = measure_2d_sparsity_per_sample(X_3d, wavelet)
    y_orig = measure_2d_sparsity_per_sample(Y_3d, wavelet)
    x_rcm = measure_2d_sparsity_per_sample(X_3d[:, :, x_perm], wavelet)
    y_rcm = measure_2d_sparsity_per_sample(Y_3d[:, :, y_perm], wavelet)

    results = {
        "Original": (x_orig, y_orig),
        "RCM cols": (x_rcm, y_rcm),
    }

    # Header
    header = f"  {'':>10s}"
    for cfg in configs:
        header += f" | {'X ' + cfg:>16s}  {'Y ' + cfg:>16s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for t in ENERGY_THRESHOLDS:
        row = f"  {fmt_energy(t):>10s}"
        for cfg in configs:
            xr, yr = results[cfg]
            xm = xr[t]["mean_pct"]
            xs = xr[t]["std_pct"]
            ym = yr[t]["mean_pct"]
            ys = yr[t]["std_pct"]
            row += f" | {xm:7.2f}%±{xs:4.2f}%  {ym:7.2f}%±{ys:4.2f}%"
        print(row)

    # Effective rank
    X_2d_cpu = X_2d.float().cpu()
    Y_2d_cpu = Y_2d.float().cpu()
    _, Sx, _ = torch.linalg.svd(X_2d_cpu, full_matrices=False)
    _, Sy, _ = torch.linalg.svd(Y_2d_cpu, full_matrices=False)
    er_x = _effective_rank(Sx)
    er_y = _effective_rank(Sy)
    print(
        f"\n  Feature ER:  X = {er_x:.1f} / {hidden} ({100*er_x/hidden:.1f}%),  Y = {er_y:.1f} / {out_hidden} ({100*er_y/out_hidden:.1f}%)"
    )

    # SVD + 1D Wavelet Hybrid
    print(f"\n  SVD + 1D Wavelet Hybrid (Y activations)")
    print(f"  {'─'*100}")

    hybrid_results, hybrid_er, hybrid_ranks = measure_svd_wavelet_hybrid(Y_3d, wavelet)

    # Print SVD rank info
    for r in hybrid_ranks:
        hr = hybrid_results[r]
        print(
            f"  rank={r:>4d}:  SVD captures {hr['svd_retention']:6.2f}% energy,  SVD-only cost = {hr['svd_only_pct']:.2f}%"
        )

    # Comparison table: total params as % of original (seq_len × out_hidden)
    hdr = f"  {'':>12} | {'2D Wav RCM':>12}"
    for r in hybrid_ranks:
        svd_e = hybrid_results[r]["svd_retention"]
        hdr += f" | r={r}({svd_e:.1f}%E)"
    print(f"\n{hdr}")
    print(f"  {'─' * len(hdr)}")

    for t in ENERGY_THRESHOLDS:
        row = f"  {fmt_energy(t):>12} | {y_rcm[t]['mean_pct']:11.2f}%"
        for r in hybrid_ranks:
            s = hybrid_results[r]["summary"][t]
            if s["achievable"]:
                row += f" | {s['hybrid_pct_mean']:11.2f}%"
            else:
                row += f" |          n/a"
        print(row)


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

    wavelet = "db2"

    for layer_name in LAYER_NAMES:
        print(f"\n{'#'*160}")
        print(f"# Layer: {layer_name}")
        print(f"{'#'*160}")

        print(f"\nCollecting activations for {layer_name}...")
        X_3d = collect_activations(model, layer_name, dataloader)
        print(f"X (3D): {X_3d.shape}")

        parts = layer_name.split(".")
        layer = model
        for part in parts:
            layer = layer[int(part)] if part.isdigit() else getattr(layer, part)

        W = layer.weight.data.float().cpu()
        print_table(layer_name, X_3d, W, wavelet)


if __name__ == "__main__":
    main()
