"""Wavelet basis analysis: transform activations along sequence dimension,
analyze per-band activation structure and temporal truncation quality."""

import torch
import numpy as np
import pywt
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/brian-dellabetta/projects/llm-compressor")
from wavelet_cascade import _effective_rank
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

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
WAVELET = "db2"
MODE = "periodization"
# The level is an implementation knob, not a compute knob. Lower levels (4) mean fewer
# bands and fewer kernel launches — simpler to implement. Higher levels (8-9) give
# better analytical resolution of where the energy lives but don't change the total
# adaptive-rank compute budget. For a practical per-band adaptive-rank system, level
# 4 with 5 bands is probably the sweet spot — simple, and the A4 band already achieves
# ER=3-4 for the projector layers.
SEQ_LEVEL = 4
ER_SUBSAMPLE = 4096
FEAT_LEVEL = 4

# TODO try this on a decoder layer itself, not just individual linear layers.
# Alternatively, you could collect Y as the output after a nonlinearity
# (e.g., map pre-attention residual → post-attention residual), which would break the
# exact linear relationship and let lstsq find something genuinely different.


def get_band_structure(n, wavelet=WAVELET, level=None):
    if level is None:
        max_level = pywt.dwt_max_level(n, wavelet)
        level = max_level
        while level > 0:
            coeffs = pywt.wavedec(np.zeros(n), wavelet, level=level, mode=MODE)
            if sum(len(c) for c in coeffs) == n:
                break
            level -= 1

    coeffs = pywt.wavedec(np.zeros(n), wavelet, level=level, mode=MODE)
    bands = []
    offset = 0
    for i, c in enumerate(coeffs):
        label = f"A{level}" if i == 0 else f"D{level - i + 1}"
        bands.append(
            {"label": label, "start": offset, "end": offset + len(c), "size": len(c)}
        )
        offset += len(c)
    return bands, level


def transform_sequence(X_3d, wavelet=WAVELET, level=None):
    _, seq_len, _ = X_3d.shape
    if level is None:
        _, level = get_band_structure(seq_len, wavelet)
    X_np = X_3d.numpy() if isinstance(X_3d, torch.Tensor) else X_3d
    coeffs = pywt.wavedec(X_np, wavelet, level=level, mode=MODE, axis=1)
    return np.concatenate(coeffs, axis=1)


def inverse_transform_sequence(X_tilde_3d, wavelet=WAVELET, level=None, seq_len=None):
    if seq_len is None:
        seq_len = X_tilde_3d.shape[1]
    if level is None:
        _, level = get_band_structure(seq_len, wavelet)
    bands, _ = get_band_structure(seq_len, wavelet)
    coeffs = []
    offset = 0
    for b in bands:
        coeffs.append(X_tilde_3d[:, offset : offset + b["size"], :])
        offset += b["size"]
    return pywt.waverec(coeffs, wavelet, mode=MODE, axis=1)[:, :seq_len, :]


def transform_features(X_2d, wavelet=WAVELET, level=None):
    _, n_features = X_2d.shape
    if level is None:
        _, level = get_band_structure(n_features, wavelet)
    X_np = X_2d.numpy() if isinstance(X_2d, torch.Tensor) else X_2d
    coeffs = pywt.wavedec(X_np, wavelet, level=level, mode=MODE, axis=1)
    return np.concatenate(coeffs, axis=1)


def measure_snr(Y_true, Y_approx):
    if isinstance(Y_true, np.ndarray):
        Y_true = torch.from_numpy(Y_true).float()
    if isinstance(Y_approx, np.ndarray):
        Y_approx = torch.from_numpy(Y_approx).float()
    signal_power = torch.var(Y_true)
    mse = torch.mean((Y_true - Y_approx) ** 2)
    snr = 10 * torch.log10(signal_power / (mse + 1e-10)).item()
    return snr


def collect_activations(model, layer_name, dataloader, device="cuda"):
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


def analyze_band(X_band_2d, Y_band_2d, in_features, out_features):
    n_vectors = X_band_2d.shape[0]
    subsample = min(n_vectors, ER_SUBSAMPLE)
    idx = (
        np.random.choice(n_vectors, subsample, replace=False)
        if n_vectors > subsample
        else np.arange(n_vectors)
    )

    X_sub = torch.from_numpy(X_band_2d[idx]).float()
    _, Sx, _ = torch.linalg.svd(X_sub, full_matrices=False)
    er_x = _effective_rank(Sx)

    Y_sub = torch.from_numpy(Y_band_2d[idx]).float()
    _, Sy, _ = torch.linalg.svd(Y_sub, full_matrices=False)
    er_y = _effective_rank(Sy)

    return er_x, er_y


def analyze_wopt_structure(W, wavelet, layer_name):
    """Compute W_opt = Ψ_out W Ψ_in^T (feature-dim wavelet basis) and visualize."""
    d_out, d_in = W.shape
    W_np = W.cpu().float().numpy()

    in_bands, in_level = get_band_structure(d_in, wavelet)
    out_bands, out_level = get_band_structure(d_out, wavelet)

    coeffs_cols = pywt.wavedec(W_np, wavelet, level=in_level, mode=MODE, axis=1)
    W_step1 = np.concatenate(coeffs_cols, axis=1)
    coeffs_rows = pywt.wavedec(W_step1, wavelet, level=out_level, mode=MODE, axis=0)
    W_opt = np.concatenate(coeffs_rows, axis=0).astype(np.float64)

    total_energy = np.sum(W_opt**2)
    total_elements = d_out * d_in

    n_ob, n_ib = len(out_bands), len(in_bands)
    block_energy = np.zeros((n_ob, n_ib))
    block_density = np.zeros((n_ob, n_ib))

    for i, ob in enumerate(out_bands):
        for j, ib in enumerate(in_bands):
            blk = W_opt[ob["start"] : ob["end"], ib["start"] : ib["end"]]
            be = np.sum(blk**2)
            block_energy[i, j] = be / total_energy * 100
            expected = ob["size"] * ib["size"] / total_elements * 100
            block_density[i, j] = block_energy[i, j] / expected if expected > 0 else 0

    in_parts = " ".join(f"{b['label']}[{b['size']}]" for b in in_bands)
    out_parts = " ".join(f"{b['label']}[{b['size']}]" for b in out_bands)
    print(f"\n  Feature-dim W_opt = Ψ_out W Ψ_in^T")
    print(f"    Input bands  (level={in_level}): {in_parts}")
    print(f"    Output bands (level={out_level}): {out_parts}")

    print(f"\n  Block energy (% of ||W_opt||²):")
    header = f"  {'out\\in':>6}"
    for ib in in_bands:
        header += f" {ib['label']:>7}"
    header += " |   row%"
    print(header)
    print(f"  {'─' * (8 + 8 * n_ib + 9)}")

    for i, ob in enumerate(out_bands):
        row = f"  {ob['label']:>6}"
        row_total = 0
        for j in range(n_ib):
            e = block_energy[i, j]
            row_total += e
            if e >= 1.0:
                row += f"  {e:5.1f}%"
            elif e >= 0.01:
                row += f"  {e:5.2f}%"
            else:
                row += f"      ─"
        row += f" | {row_total:5.1f}%"
        print(row)

    diag_energy = sum(block_energy[i, i] for i in range(min(n_ob, n_ib)))
    print(f"\n  Diagonal: {diag_energy:.1f}%  Off-diagonal: {100 - diag_energy:.1f}%")

    print(f"\n  Energy density (× uniform baseline, >1 = denser than random):")
    header2 = f"  {'out\\in':>6}"
    for ib in in_bands:
        header2 += f" {ib['label']:>7}"
    print(header2)
    print(f"  {'─' * (8 + 8 * n_ib)}")

    for i, ob in enumerate(out_bands):
        row = f"  {ob['label']:>6}"
        for j in range(n_ib):
            d = block_density[i, j]
            if d >= 10:
                row += f"   {d:4.0f}×"
            elif d >= 1:
                row += f"   {d:4.1f}×"
            else:
                row += f"   {d:4.2f}×"
        print(row)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    ax = axes[0]
    log_mag = np.log10(np.abs(W_opt).astype(np.float64) + 1e-10)
    im = ax.imshow(log_mag, cmap="viridis", aspect="auto", interpolation="none")
    for b in in_bands[:-1]:
        ax.axvline(b["end"] - 0.5, color="red", linewidth=0.5, alpha=0.6)
    for b in out_bands[:-1]:
        ax.axhline(b["end"] - 0.5, color="red", linewidth=0.5, alpha=0.6)
    ax.set_title("log₁₀|W_opt|")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    im2 = ax.imshow(block_energy, cmap="hot_r", aspect="equal", interpolation="none")
    ax.set_xticks(range(n_ib))
    ax.set_xticklabels(
        [b["label"] for b in in_bands], rotation=45, ha="right", fontsize=7
    )
    ax.set_yticks(range(n_ob))
    ax.set_yticklabels([b["label"] for b in out_bands], fontsize=7)
    ax.set_title("Block Energy (%)")
    for i in range(n_ob):
        for j in range(n_ib):
            e = block_energy[i, j]
            if e >= 0.05:
                color = "white" if e > 5 else "black"
                txt = f"{e:.1f}" if e >= 1 else f"{e:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5, color=color)
    plt.colorbar(im2, ax=ax, shrink=0.8)

    ax = axes[2]
    max_dist = max(n_ob, n_ib)
    cum_by_dist = np.zeros(max_dist)
    for dist in range(max_dist):
        for i in range(n_ob):
            for j in range(n_ib):
                if abs(i - j) == dist:
                    cum_by_dist[dist] += block_energy[i, j]
    ax.bar(range(max_dist), cum_by_dist, color="steelblue")
    ax.set_xlabel("Block distance from diagonal")
    ax.set_ylabel("Energy (%)")
    ax.set_title("Energy vs diagonal distance")
    cum = np.cumsum(cum_by_dist)
    ax2 = ax.twinx()
    ax2.plot(range(max_dist), cum, "r-o", markersize=3)
    ax2.set_ylabel("Cumulative %", color="red")
    ax2.set_ylim(0, 105)

    plt.suptitle(f"{layer_name}  W_opt ({d_out}×{d_in})", fontsize=11)
    plt.tight_layout()

    safe_name = layer_name.replace(".", "_")
    plot_path = f"/home/brian-dellabetta/projects/llm-compressor/wopt_{safe_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {plot_path}")


def analyze_ridge(X_3d, W, wavelet, layer_name, device="cuda"):
    """Ridge regression: W̃ = argmin ||Ỹ - X̃W̃^T||² + λ||W̃||²
    where X̃, Ỹ are feature-dimension wavelet transforms."""
    n_samples, seq_len, in_features = X_3d.shape
    out_features = W.shape[0]
    N = n_samples * seq_len

    X_np = X_3d.numpy() if isinstance(X_3d, torch.Tensor) else X_3d
    X_2d = X_np.reshape(-1, in_features)

    in_bands, in_level = get_band_structure(in_features, wavelet)
    out_bands, out_level = get_band_structure(out_features, wavelet)

    in_parts = " ".join(f"{b['label']}[{b['size']}]" for b in in_bands)
    out_parts = " ".join(f"{b['label']}[{b['size']}]" for b in out_bands)
    print(f"    Input bands  (level={in_level}): {in_parts}")
    print(f"    Output bands (level={out_level}): {out_parts}")

    W_gpu = W.float().to(device)

    G = torch.zeros((in_features, in_features), dtype=torch.float32, device=device)
    B = torch.zeros((in_features, out_features), dtype=torch.float32, device=device)
    Y_sq_sum = 0.0

    chunk_size = 8192
    n_chunks = (N + chunk_size - 1) // chunk_size
    print(f"    Computing Gram matrix ({n_chunks} chunks, N={N})...")

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, N)
        X_chunk = X_2d[start:end].astype(np.float32)

        X_gpu = torch.from_numpy(X_chunk).to(device)
        Y_chunk = (X_gpu @ W_gpu.T).cpu().numpy()
        del X_gpu

        Xt_chunk = transform_features(X_chunk, wavelet, in_level)
        Yt_chunk = transform_features(Y_chunk, wavelet, out_level)

        Xt = torch.from_numpy(Xt_chunk.astype(np.float32)).to(device)
        Yt = torch.from_numpy(Yt_chunk.astype(np.float32)).to(device)
        G += Xt.T @ Xt
        B += Xt.T @ Yt
        Y_sq_sum += (Yt.double() ** 2).sum().item()
        del Xt, Yt

    del W_gpu
    torch.cuda.empty_cache()
    Y_mean_sq = Y_sq_sum / (N * out_features)

    print(f"    Eigendecomposition ({in_features}×{in_features})...")
    eigenvalues, V = torch.linalg.eigh(G)
    eigenvalues = eigenvalues.flip(0)
    V = V.flip(1)
    del G

    P = V.T @ B
    q = (P.double() ** 2).sum(dim=1)
    del B

    S_w_orig = torch.linalg.svdvals(W.float().to(device))
    W_er_orig = _effective_rank(S_w_orig.cpu())
    W_energy = (W.float() ** 2).sum().item()

    eig_pos = eigenvalues[eigenvalues > 0]
    eig_min = eig_pos.min().item()
    eig_max = eig_pos.max().item()

    lambdas = np.logspace(np.log10(eig_min * 0.1), np.log10(eig_max * 10), 30)

    print(f"\n    Eigenvalue range: [{eig_min:.2e}, {eig_max:.2e}]")
    print(
        f"    Original W: ER={W_er_orig:.1f}/{min(in_features, out_features)}, energy={W_energy:.2e}"
    )
    print(f"\n    {'λ':>12} | {'SNR (dB)':>10} | {'ER(W̃)':>12}")
    print(f"    {'─' * 42}")

    results = []
    eig64 = eigenvalues.double()

    for lam in lambdas:
        D64 = 1.0 / (eig64 + lam)
        term2 = 2.0 * (D64 * q).sum().item()
        term3 = (eig64 * D64**2 * q).sum().item()
        loss = max(Y_sq_sum - term2 + term3, 0.0)

        mse = loss / (N * out_features)
        snr = 10 * np.log10(Y_mean_sq / mse) if mse > 1e-30 else 200.0

        D_f = (1.0 / (eigenvalues + lam)).unsqueeze(1)
        W_tilde_T = V @ (D_f * P)
        S_w = torch.linalg.svdvals(W_tilde_T)
        er = _effective_rank(S_w.cpu())

        results.append({"lambda": lam, "snr": snr, "er": er})

        snr_str = f"{snr:.1f}" if snr < 200 else ">200"
        print(f"    {lam:12.2e} | {snr_str:>10} | {er:>10.1f}")

    del V, P, eigenvalues, q, eig64
    torch.cuda.empty_cache()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    lams_p = [r["lambda"] for r in results]
    snrs_p = [r["snr"] for r in results]
    ers_p = [r["er"] for r in results]

    ax1.semilogx(lams_p, snrs_p, "b-o", markersize=3)
    ax1.axhline(y=40, color="r", linestyle="--", alpha=0.5, label="40dB target")
    ax1.set_xlabel("λ (regularization)")
    ax1.set_ylabel("SNR (dB)")
    ax1.set_title("Reconstruction SNR")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(lams_p, ers_p, "g-o", markersize=3)
    ax2.axhline(
        y=W_er_orig,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Original W ER={W_er_orig:.0f}",
    )
    ax2.set_xlabel("λ (regularization)")
    ax2.set_ylabel("Effective Rank")
    ax2.set_title("W̃ effective rank")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"{layer_name}  Ridge: W̃ = argmin ||Ỹ-X̃W̃ᵀ||²+λ||W̃||²", fontsize=11
    )
    plt.tight_layout()

    safe_name = layer_name.replace(".", "_")
    plot_path = f"/home/brian-dellabetta/projects/llm-compressor/ridge_{safe_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n    Plot saved: {plot_path}")


def analyze_block_rank(W, wavelet, layer_name, device="cuda"):
    """Per-block ER of W̃ = Ψ_out W Ψ_in^T at coarse wavelet level.
    Tests whether distant frequency blocks have lower rank than nearby ones."""
    d_out, d_in = W.shape
    W_np = W.cpu().float().numpy()

    in_bands, in_level = get_band_structure(d_in, wavelet, level=FEAT_LEVEL)
    out_bands, out_level = get_band_structure(d_out, wavelet, level=FEAT_LEVEL)

    coeffs_cols = pywt.wavedec(W_np, wavelet, level=in_level, mode=MODE, axis=1)
    W_step1 = np.concatenate(coeffs_cols, axis=1)
    coeffs_rows = pywt.wavedec(W_step1, wavelet, level=out_level, mode=MODE, axis=0)
    W_opt = np.concatenate(coeffs_rows, axis=0)

    total_energy = np.sum(W_opt.astype(np.float64) ** 2)

    n_ob = len(out_bands)
    n_ib = len(in_bands)

    in_parts = " ".join(f"{b['label']}[{b['size']}]" for b in in_bands)
    out_parts = " ".join(f"{b['label']}[{b['size']}]" for b in out_bands)
    print(f"    Input bands  (level={in_level}): {in_parts}")
    print(f"    Output bands (level={out_level}): {out_parts}")

    block_er = np.zeros((n_ob, n_ib))
    block_max_rank = np.zeros((n_ob, n_ib), dtype=int)
    block_energy_pct = np.zeros((n_ob, n_ib))

    for i, ob in enumerate(out_bands):
        for j, ib in enumerate(in_bands):
            blk = W_opt[ob["start"] : ob["end"], ib["start"] : ib["end"]]
            blk_t = torch.from_numpy(blk.astype(np.float32)).to(device)
            S = torch.linalg.svdvals(blk_t)
            block_er[i, j] = _effective_rank(S.cpu())
            block_max_rank[i, j] = min(ob["size"], ib["size"])
            block_energy_pct[i, j] = (
                np.sum(blk.astype(np.float64) ** 2) / total_energy * 100
            )

    print(f"\n    Block ER / max_rank:")
    header = f"    {'out\\in':>6}"
    for ib in in_bands:
        header += f" {ib['label']:>12}"
    print(header)
    print(f"    {'─' * (8 + 13 * n_ib)}")

    for i, ob in enumerate(out_bands):
        row = f"    {ob['label']:>6}"
        for j in range(n_ib):
            er = block_er[i, j]
            mr = block_max_rank[i, j]
            row += f"  {er:>5.0f}/{mr:<5d}"
        print(row)

    print(f"\n    Block ER as % of max rank:")
    header2 = f"    {'out\\in':>6}"
    for ib in in_bands:
        header2 += f" {ib['label']:>7}"
    print(header2)
    print(f"    {'─' * (8 + 8 * n_ib)}")

    for i, ob in enumerate(out_bands):
        row = f"    {ob['label']:>6}"
        for j in range(n_ib):
            pct = 100 * block_er[i, j] / block_max_rank[i, j]
            row += f"  {pct:5.1f}%"
        print(row)

    print(f"\n    Average ER% by band distance:")
    max_dist = max(n_ob, n_ib)
    dist_avg = []
    for dist in range(max_dist):
        pcts = []
        for i in range(n_ob):
            for j in range(n_ib):
                if abs(i - j) == dist:
                    pcts.append(100 * block_er[i, j] / block_max_rank[i, j])
        if pcts:
            avg = np.mean(pcts)
            dist_avg.append(avg)
            print(f"    dist={dist}: {avg:5.1f}%  (n={len(pcts)} blocks)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    er_pct = block_er / block_max_rank * 100
    im = ax.imshow(er_pct, cmap="RdYlGn_r", aspect="equal", interpolation="none")
    ax.set_xticks(range(n_ib))
    ax.set_xticklabels([b["label"] for b in in_bands], rotation=45, ha="right")
    ax.set_yticks(range(n_ob))
    ax.set_yticklabels([b["label"] for b in out_bands])
    ax.set_title("Block ER as % of max rank")
    for i in range(n_ob):
        for j in range(n_ib):
            ax.text(
                j, i, f"{er_pct[i, j]:.0f}%",
                ha="center", va="center", fontsize=8,
                color="white" if er_pct[i, j] > 50 else "black",
            )
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    ax.bar(range(len(dist_avg)), dist_avg, color="steelblue")
    ax.set_xlabel("Band distance")
    ax.set_ylabel("Avg ER % of max rank")
    ax.set_title("ER% vs band distance")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"{layer_name}  Block rank analysis (level={FEAT_LEVEL})", fontsize=11
    )
    plt.tight_layout()

    safe_name = layer_name.replace(".", "_")
    plot_path = (
        f"/home/brian-dellabetta/projects/llm-compressor/blockrank_{safe_name}.png"
    )
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n    Plot saved: {plot_path}")


def _entanglement_from_sv(S, max_rank):
    """Compute entropy, S/Smax, and Schmidt ranks from singular values."""
    energy_thresholds = [0.90, 0.99, 0.999, 0.9999]
    S = S[S > 1e-10]
    S_sorted, _ = torch.sort(S, descending=True)
    sv2 = S_sorted.double() ** 2
    total_energy = sv2.sum().item()
    if total_energy < 1e-30:
        return {
            "entropy": 0.0, "max_entropy": 0.0, "s_ratio": 0.0,
            "schmidt_ranks": {t: 0 for t in energy_thresholds},
            "max_rank": max_rank,
        }
    p = sv2 / sv2.sum()
    entropy = -(p * torch.log(p + 1e-30)).sum().item()
    max_entropy = np.log(max_rank) if max_rank > 0 else 0.0
    cum_energy = torch.cumsum(sv2, dim=0)
    schmidt_ranks = {}
    for t in energy_thresholds:
        target = t * total_energy
        idx = (cum_energy >= target).nonzero(as_tuple=True)[0]
        schmidt_ranks[t] = idx[0].item() + 1 if len(idx) > 0 else len(S_sorted)
    return {
        "entropy": entropy, "max_entropy": max_entropy,
        "s_ratio": entropy / max_entropy if max_entropy > 0 else 0,
        "schmidt_ranks": schmidt_ranks, "max_rank": len(S_sorted),
    }


def _w_cross_cut_sv(W_matrix, out_bands, in_bands, cut, device="cuda"):
    """SVD of cross-cut blocks of a weight matrix at a hierarchical cut."""
    coarse_out_idx = []
    for i in range(cut):
        coarse_out_idx.extend(range(out_bands[i]["start"], out_bands[i]["end"]))
    fine_out_idx = []
    for i in range(cut, len(out_bands)):
        fine_out_idx.extend(range(out_bands[i]["start"], out_bands[i]["end"]))
    coarse_in_idx = []
    for i in range(cut):
        coarse_in_idx.extend(range(in_bands[i]["start"], in_bands[i]["end"]))
    fine_in_idx = []
    for i in range(cut, len(in_bands)):
        fine_in_idx.extend(range(in_bands[i]["start"], in_bands[i]["end"]))

    cross_upper = W_matrix[np.ix_(coarse_out_idx, fine_in_idx)]
    cross_lower = W_matrix[np.ix_(fine_out_idx, coarse_in_idx)]

    all_sv = []
    for block in [cross_upper, cross_lower]:
        blk_t = torch.from_numpy(block.astype(np.float32)).to(device)
        S = torch.linalg.svdvals(blk_t).cpu()
        all_sv.append(S)

    S_combined = torch.cat(all_sv)
    max_rank = min(len(coarse_out_idx), len(fine_in_idx)) + min(len(fine_out_idx), len(coarse_in_idx))
    return S_combined, max_rank


def _activation_cross_cov_sv(X_tilde_np, bands, cut, device="cuda"):
    """SVD of cross-covariance between coarse and fine feature bands of activations."""
    coarse_idx = []
    for i in range(cut):
        coarse_idx.extend(range(bands[i]["start"], bands[i]["end"]))
    fine_idx = []
    for i in range(cut, len(bands)):
        fine_idx.extend(range(bands[i]["start"], bands[i]["end"]))

    X_coarse = X_tilde_np[:, coarse_idx]
    X_fine = X_tilde_np[:, fine_idx]
    C = X_coarse.T @ X_fine
    C_t = torch.from_numpy(C.astype(np.float32)).to(device)
    S = torch.linalg.svdvals(C_t).cpu()
    max_rank = min(len(coarse_idx), len(fine_idx))
    return S, max_rank


def analyze_mera_combined(W, X_3d, wavelet, layer_name, device="cuda"):
    """Combined MERA viability: X, Y, W (basis change), W_seq (sequential lstsq)."""
    d_out, d_in = W.shape
    n_samples, seq_len, _ = X_3d.shape
    n_tokens = n_samples * seq_len

    in_bands, in_level = get_band_structure(d_in, wavelet, level=FEAT_LEVEL)
    out_bands, out_level = get_band_structure(d_out, wavelet, level=FEAT_LEVEL)
    n_bands = len(in_bands)

    in_parts = " ".join(f"{b['label']}[{b['size']}]" for b in in_bands)
    out_parts = " ".join(f"{b['label']}[{b['size']}]" for b in out_bands)
    print(f"    Input bands  (level={in_level}): {in_parts}")
    print(f"    Output bands (level={out_level}): {out_parts}")

    # Flatten activations and compute Y on GPU in chunks
    X_2d = X_3d.reshape(-1, d_in).float()
    print(f"    Computing Y = X @ W^T on GPU...", end="", flush=True)
    W_gpu = W.to(device)
    chunk_size = 16384
    Y_chunks = []
    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        Y_chunks.append((X_2d[start:end].to(device) @ W_gpu.T).cpu())
    Y_2d = torch.cat(Y_chunks, dim=0)
    del W_gpu, Y_chunks
    torch.cuda.empty_cache()
    print(f" done. Y: {Y_2d.shape}")

    # Wavelet transform features
    print(f"    Wavelet-transforming X features...", end="", flush=True)
    X_tilde = transform_features(X_2d.cpu(), wavelet, level=in_level)
    print(f" done. Ỹ...", end="", flush=True)
    Y_tilde = transform_features(Y_2d, wavelet, level=out_level)
    print(f" done.")
    del X_2d, Y_2d

    # W̃ = Ψ_out W Ψ_in^T (basis change)
    W_np = W.cpu().float().numpy()
    coeffs_cols = pywt.wavedec(W_np, wavelet, level=in_level, mode=MODE, axis=1)
    W_step1 = np.concatenate(coeffs_cols, axis=1)
    coeffs_rows = pywt.wavedec(W_step1, wavelet, level=out_level, mode=MODE, axis=0)
    W_tilde = np.concatenate(coeffs_rows, axis=0)

    # W̃_seq = sequential lstsq (diagonal first, then off-diagonal on residual)
    print(f"    Sequential lstsq...", end="", flush=True)
    W_seq = np.zeros((d_out, d_in), dtype=np.float64)
    diag_Y_approx = np.zeros_like(Y_tilde)

    for i in range(n_bands):
        ob = out_bands[i]
        ib = in_bands[i]
        X_i = X_tilde[:, ib["start"]:ib["end"]].astype(np.float64)
        Y_i = Y_tilde[:, ob["start"]:ob["end"]].astype(np.float64)
        XtX = X_i.T @ X_i
        XtY = X_i.T @ Y_i
        W_ii_T = np.linalg.solve(XtX, XtY)
        W_seq[ob["start"]:ob["end"], ib["start"]:ib["end"]] = W_ii_T.T
        diag_Y_approx[:, ob["start"]:ob["end"]] = X_i @ W_ii_T

    diag_snr = measure_snr(Y_tilde, diag_Y_approx)
    print(f" diag SNR={diag_snr:.1f}dB.", end="", flush=True)

    # Off-diagonal: for each output band, regress residual on all other input bands
    for i in range(n_bands):
        ob = out_bands[i]
        ib = in_bands[i]
        Y_i = Y_tilde[:, ob["start"]:ob["end"]].astype(np.float64)
        R_i = Y_i - diag_Y_approx[:, ob["start"]:ob["end"]]

        other_in_ranges = []
        for j in range(n_bands):
            if j != i:
                jb = in_bands[j]
                other_in_ranges.append((j, jb))

        other_cols = np.concatenate(
            [X_tilde[:, jb["start"]:jb["end"]] for _, jb in other_in_ranges],
            axis=1,
        ).astype(np.float64)

        XoXo = other_cols.T @ other_cols
        XoR = other_cols.T @ R_i
        W_off_T = np.linalg.solve(XoXo, XoR)

        offset = 0
        for j, jb in other_in_ranges:
            sz = jb["size"]
            W_seq[ob["start"]:ob["end"], jb["start"]:jb["end"]] = W_off_T[offset:offset + sz].T
            offset += sz

    full_Y_approx = X_tilde.astype(np.float64) @ W_seq.T
    full_snr = measure_snr(Y_tilde, full_Y_approx)
    print(f" full SNR={full_snr:.1f}dB. done.")
    del diag_Y_approx, full_Y_approx

    # Entanglement analysis at each hierarchical cut
    analyses = ["X", "Y", "W", "W_seq"]
    cut_data = {a: [] for a in analyses}

    for cut in range(1, n_bands):
        coarse_labels = ",".join(in_bands[i]["label"] for i in range(cut))
        fine_labels = ",".join(in_bands[i]["label"] for i in range(cut, n_bands))

        # X: cross-covariance SVD
        S_x, mr_x = _activation_cross_cov_sv(X_tilde, in_bands, cut, device)
        cut_data["X"].append({"cut": cut, **_entanglement_from_sv(S_x, mr_x),
                              "coarse": coarse_labels, "fine": fine_labels})

        # Y: cross-covariance SVD
        S_y, mr_y = _activation_cross_cov_sv(Y_tilde, out_bands, cut, device)
        cut_data["Y"].append({"cut": cut, **_entanglement_from_sv(S_y, mr_y),
                              "coarse": coarse_labels, "fine": fine_labels})

        # W: cross-cut SVD
        S_w, mr_w = _w_cross_cut_sv(W_tilde, out_bands, in_bands, cut, device)
        cut_data["W"].append({"cut": cut, **_entanglement_from_sv(S_w, mr_w),
                              "coarse": coarse_labels, "fine": fine_labels})

        # W_seq: cross-cut SVD
        S_ws, mr_ws = _w_cross_cut_sv(W_seq, out_bands, in_bands, cut, device)
        cut_data["W_seq"].append({"cut": cut, **_entanglement_from_sv(S_ws, mr_ws),
                                  "coarse": coarse_labels, "fine": fine_labels})

    del X_tilde, Y_tilde

    # Print combined tables
    print(f"\n    Diagonal-only lstsq SNR: {diag_snr:.1f} dB,  Full sequential SNR: {full_snr:.1f} dB")

    print(f"\n    S/Smax (%):")
    hdr = f"    {'Cut':>4}  {'Coarse':<16} {'Fine':<16}"
    for a in analyses:
        hdr += f"  {a:>8}"
    print(hdr)
    print(f"    {'─' * (42 + 10 * len(analyses))}")
    for ci in range(n_bands - 1):
        r0 = cut_data[analyses[0]][ci]
        row = f"    {r0['cut']:>4}  {{" + r0['coarse'] + "}" + " " * max(0, 14 - len(r0['coarse']))
        row += " {" + r0['fine'] + "}" + " " * max(0, 14 - len(r0['fine']))
        for a in analyses:
            row += f"  {cut_data[a][ci]['s_ratio'] * 100:7.1f}%"
        print(row)

    print(f"\n    Schmidt rank at 99.99% energy (% of max_rank):")
    hdr2 = f"    {'Cut':>4}"
    for a in analyses:
        hdr2 += f"  {a:>12}"
    print(hdr2)
    print(f"    {'─' * (6 + 14 * len(analyses))}")
    for ci in range(n_bands - 1):
        cut_num = cut_data[analyses[0]][ci]["cut"]
        row = f"    {cut_num:>4}"
        for a in analyses:
            d = cut_data[a][ci]
            sr = d["schmidt_ranks"][0.9999]
            mr = d["max_rank"]
            pct = 100 * sr / mr if mr > 0 else 0
            row += f"  {sr:>5}/{mr:<5} {pct:4.0f}%"
        print(row)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = {"X": "tab:blue", "Y": "tab:orange", "W": "tab:red", "W_seq": "tab:green"}

    ax = axes[0]
    n_cuts = n_bands - 1
    width = 0.18
    for ai, a in enumerate(analyses):
        vals = [cut_data[a][ci]["s_ratio"] * 100 for ci in range(n_cuts)]
        positions = [ci + (ai - 1.5) * width for ci in range(n_cuts)]
        ax.bar(positions, vals, width, label=a, color=colors[a])
    ax.set_xticks(range(n_cuts))
    ax.set_xticklabels([str(i + 1) for i in range(n_cuts)])
    ax.set_xlabel("Cut level")
    ax.set_ylabel("S / S_max (%)")
    ax.set_title("Normalized entanglement entropy")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    for a in analyses:
        vals = []
        for ci in range(n_cuts):
            d = cut_data[a][ci]
            sr = d["schmidt_ranks"][0.9999]
            mr = d["max_rank"]
            vals.append(100 * sr / mr if mr > 0 else 0)
        ax.plot(range(n_cuts), vals, "o-", label=a, color=colors[a])
    ax.set_xticks(range(n_cuts))
    ax.set_xticklabels([str(i + 1) for i in range(n_cuts)])
    ax.set_xlabel("Cut level")
    ax.set_ylabel("Schmidt rank at 99.99% / max (%)")
    ax.set_title("Bond dimension needed for 40dB")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"{layer_name}  MERA combined (level={FEAT_LEVEL})", fontsize=11)
    plt.tight_layout()

    safe_name = layer_name.replace(".", "_")
    plot_path = f"/home/brian-dellabetta/projects/llm-compressor/mera_combined_{safe_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n    Plot saved: {plot_path}")


def _gram_eigenvalues(M_np, device="cuda", chunk_size=16384):
    """Compute sorted eigenvalues of M^T @ M via chunked GPU matmul + CPU eigh.

    M_np: (n_tokens, d) numpy array. Returns eigenvalues in descending order as torch tensor.
    """
    d = M_np.shape[1]
    n = M_np.shape[0]
    G = torch.zeros(d, d, dtype=torch.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = torch.from_numpy(M_np[start:end].astype(np.float64)).to(device)
        G += (chunk.T @ chunk).cpu()
    eigvals = torch.linalg.eigvalsh(G)
    return torch.flip(eigvals, [0]).clamp(min=0)


def _rank_at_energy(eigvals, thresholds):
    """Given descending eigenvalues, find rank needed for each energy threshold."""
    total = eigvals.sum().item()
    if total < 1e-30:
        return {t: 0 for t in thresholds}
    cum = torch.cumsum(eigvals, dim=0)
    ranks = {}
    for t in thresholds:
        target = t * total
        idx = (cum >= target).nonzero(as_tuple=True)[0]
        ranks[t] = idx[0].item() + 1 if len(idx) > 0 else len(eigvals)
    return ranks


def analyze_mera_decomposition(X_3d, W, wavelet, layer_name, device="cuda"):
    """Per-band SVD rank analysis: how much can MERA compress X and Y activations?"""
    d_out, d_in = W.shape
    n_samples, seq_len, _ = X_3d.shape
    n_tokens = n_samples * seq_len

    in_bands, in_level = get_band_structure(d_in, wavelet, level=FEAT_LEVEL)
    out_bands, out_level = get_band_structure(d_out, wavelet, level=FEAT_LEVEL)

    in_parts = " ".join(f"{b['label']}[{b['size']}]" for b in in_bands)
    out_parts = " ".join(f"{b['label']}[{b['size']}]" for b in out_bands)
    print(f"    Input bands  (level={in_level}): {in_parts}")
    print(f"    Output bands (level={out_level}): {out_parts}")

    X_2d = X_3d.reshape(-1, d_in).float()
    print(f"    Computing Y = X @ W^T on GPU...", end="", flush=True)
    W_gpu = W.to(device)
    chunk_size = 16384
    Y_chunks = []
    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        Y_chunks.append((X_2d[start:end].to(device) @ W_gpu.T).cpu())
    Y_2d = torch.cat(Y_chunks, dim=0)
    del W_gpu, Y_chunks
    torch.cuda.empty_cache()
    print(f" done. Y: {Y_2d.shape}")

    print(f"    Wavelet-transforming X features...", end="", flush=True)
    X_tilde = transform_features(X_2d.cpu(), wavelet, level=in_level)
    print(f" done. Ỹ...", end="", flush=True)
    Y_tilde = transform_features(Y_2d, wavelet, level=out_level)
    print(f" done.")
    del X_2d, Y_2d

    thresholds = [0.99, 0.999, 0.9999]
    threshold_labels = ["99%", "99.9%", "99.99%"]

    def analyze_one(M_tilde, bands, label):
        """Per-band and full SVD rank analysis for one activation matrix."""
        d = M_tilde.shape[1]
        total_energy = np.sum(M_tilde.astype(np.float64) ** 2)

        band_results = []
        for b in bands:
            band_data = M_tilde[:, b["start"]:b["end"]]
            band_energy = np.sum(band_data.astype(np.float64) ** 2)
            energy_pct = 100 * band_energy / total_energy if total_energy > 0 else 0

            print(f"      {label} band {b['label']} ({b['size']})...", end="", flush=True)
            eigvals = _gram_eigenvalues(band_data, device=device)
            ranks = _rank_at_energy(eigvals, thresholds)
            print(f" done.")

            band_results.append({
                "label": b["label"],
                "size": b["size"],
                "energy_pct": energy_pct,
                "ranks": ranks,
            })

        print(f"      {label} full ({d})...", end="", flush=True)
        full_eigvals = _gram_eigenvalues(M_tilde, device=device)
        full_ranks = _rank_at_energy(full_eigvals, thresholds)
        print(f" done.")

        return band_results, full_ranks

    print(f"\n    Computing per-band SVD ranks...")
    x_band_results, x_full_ranks = analyze_one(X_tilde, in_bands, "X")
    y_band_results, y_full_ranks = analyze_one(Y_tilde, out_bands, "Y")

    del X_tilde, Y_tilde

    def print_table(label, n_tok, dim, band_results, full_ranks):
        print(f"\n    {label} activations ({n_tok} × {dim}):")
        hdr = f"    {'Band':>6}  {'Size':>6}  {'Energy%':>8}"
        for tl in threshold_labels:
            hdr += f"  {'r@'+tl:>10}"
        print(hdr)
        print(f"    {'─' * (26 + 12 * len(thresholds))}")

        per_band_totals = {t: 0 for t in thresholds}
        for br in band_results:
            row = f"    {br['label']:>6}  {br['size']:>6}  {br['energy_pct']:7.2f}%"
            for t in thresholds:
                r = br["ranks"][t]
                row += f"  {r:>10}"
                per_band_totals[t] += r
            print(row)

        print(f"    {'─' * (26 + 12 * len(thresholds))}")
        row_pbt = f"    {'Total':>6}  {'':>6}  {'':>8}"
        for t in thresholds:
            row_pbt += f"  {per_band_totals[t]:>10}"
        print(row_pbt)

        row_full = f"    {'Full':>6}  {'':>6}  {'':>8}"
        for t in thresholds:
            row_full += f"  {full_ranks[t]:>10}"
        print(row_full)

        row_ratio = f"    {'Ratio':>6}  {'':>6}  {'':>8}"
        for t in thresholds:
            ratio = per_band_totals[t] / full_ranks[t] if full_ranks[t] > 0 else float("inf")
            row_ratio += f"  {ratio:>9.2f}×"
        print(row_ratio)

        row_comp = f"    {'Compr':>6}  {'':>6}  {'':>8}"
        for t in thresholds:
            comp = 100 * per_band_totals[t] / dim
            row_comp += f"  {comp:>9.1f}%"
        print(row_comp)

    print(f"\n    {'='*80}")
    print(f"    MERA Decomposition: {layer_name}  (level={FEAT_LEVEL})")
    print(f"    {'='*80}")

    print_table("X", n_tokens, d_in, x_band_results, x_full_ranks)
    print_table("Y", n_tokens, d_out, y_band_results, y_full_ranks)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    target_t = 0.99

    for ax, (label, band_results, full_ranks, dim) in zip(
        axes,
        [("X", x_band_results, x_full_ranks, d_in),
         ("Y", y_band_results, y_full_ranks, d_out)],
    ):
        band_labels = [br["label"] for br in band_results]
        band_ranks = [br["ranks"][target_t] for br in band_results]
        band_sizes = [br["size"] for br in band_results]

        x_pos = np.arange(len(band_labels))
        bars_rank = ax.bar(x_pos - 0.15, band_ranks, 0.3, label="Rank@99%", color="steelblue")
        bars_size = ax.bar(x_pos + 0.15, band_sizes, 0.3, label="Band size", color="lightcoral", alpha=0.5)

        for bar, r, s in zip(bars_rank, band_ranks, band_sizes):
            pct = 100 * r / s if s > 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{pct:.0f}%", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(band_labels)
        ax.set_ylabel("Dimensions")
        ax.set_title(f"{label}: per-band rank@99% vs band size")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        total_rank = sum(band_ranks)
        full_r = full_ranks[target_t]
        ax.text(0.98, 0.95,
                f"Sum ranks: {total_rank}/{dim} ({100*total_rank/dim:.1f}%)\n"
                f"Full SVD: {full_r}/{dim} ({100*full_r/dim:.1f}%)\n"
                f"Overhead: {total_rank/full_r:.2f}×" if full_r > 0 else f"Sum: {total_rank}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.suptitle(f"{layer_name}  MERA decomposition (level={FEAT_LEVEL})", fontsize=11)
    plt.tight_layout()

    safe_name = layer_name.replace(".", "_")
    plot_path = f"/home/brian-dellabetta/projects/llm-compressor/mera_decomp_{safe_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n    Plot saved: {plot_path}")


def fit_band_w(X_band_2d, Y_band_2d, device="cuda"):
    X_t = torch.from_numpy(X_band_2d).float().to(device)
    Y_t = torch.from_numpy(Y_band_2d).float().to(device)
    W_tilde_T = torch.linalg.lstsq(X_t, Y_t).solution
    W_tilde = W_tilde_T.T
    w_energy = (W_tilde**2).sum().item()
    S_w = torch.linalg.svdvals(W_tilde.float())
    w_er = _effective_rank(S_w)
    return w_energy, w_er


TARGET_SNRS = [30, 40, 50]


def rank_sweep_band(X_band_2d_np, W, device="cuda"):
    """For a temporal band, find minimum rank of W projected onto the band's
    OUTPUT subspace that achieves each target SNR.

    Approach: SVD of Y_band gives the output basis V_r. Then:
      W_out = W^T @ V_r   (d_in × r, precomputed offline)
      Y_approx = X @ W_out @ V_r^T
    Cost per token: r × (d_in + d_out) vs full d_in × d_out.
    """
    X_t = torch.from_numpy(X_band_2d_np).float().to(device)
    n, d_in = X_t.shape
    d_out = W.shape[0]
    W_gpu = W.to(device)

    Y_true = X_t @ W_gpu.T
    signal_energy = torch.mean(Y_true**2).item()

    _, S_y, Vt_y = torch.linalg.svd(Y_true, full_matrices=False)
    er_y = _effective_rank(S_y.cpu())
    total_sv_energy = (S_y**2).sum().item()

    max_rank = min(n, d_out)
    ranks_to_try = [
        r
        for r in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        if 1 <= r <= max_rank
    ]

    results = {t: None for t in TARGET_SNRS}
    snr_at_rank = []
    cum_sv_energy = torch.cumsum(S_y**2, dim=0)

    for r in ranks_to_try:
        energy_frac = cum_sv_energy[r - 1].item() / total_sv_energy
        V_r = Vt_y[:r, :].T
        W_out = W_gpu.T @ V_r
        Y_approx = X_t @ W_out @ V_r.T
        mse = torch.mean((Y_true - Y_approx) ** 2).item()
        if mse < 1e-20:
            snr = float("inf")
        else:
            snr = 10 * np.log10(signal_energy / mse)
        snr_at_rank.append((r, snr, energy_frac))

        for t in TARGET_SNRS:
            if results[t] is None and snr >= t:
                params = r * (d_in + d_out)
                full_params = d_in * d_out
                flops_ratio = params / full_params
                results[t] = {
                    "rank": r,
                    "params": params,
                    "flops_ratio": flops_ratio,
                    "snr": snr,
                }

    return results, snr_at_rank, er_y


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

    seq_bands, seq_level = get_band_structure(MAX_SEQUENCE_LENGTH, level=SEQ_LEVEL)
    print(
        f"\n  Temporal band structure (seq={MAX_SEQUENCE_LENGTH}, level={seq_level}):"
    )
    parts = [f"{b['label']}[{b['size']}]" for b in seq_bands]
    print(f"  {' '.join(parts)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for layer_name in LAYER_NAMES:
        print(f"\n{'#' * 120}")
        print(f"# Layer: {layer_name}")
        print(f"{'#' * 120}")

        print(f"\nCollecting activations for {layer_name}...")
        X_3d = collect_activations(model, layer_name, dataloader)
        n_samples, seq_len, in_features = X_3d.shape
        print(f"  X_3d: {X_3d.shape}")

        layer_ref = model
        for part in layer_name.split("."):
            layer_ref = (
                layer_ref[int(part)] if part.isdigit() else getattr(layer_ref, part)
            )
        W = layer_ref.weight.data.float()
        out_features = W.shape[0]
        print(f"  W: {W.shape}")

        print(f"\n  Transforming X along sequence dimension...")
        X_tilde = transform_sequence(X_3d, level=seq_level)
        print(f"  X̃: {X_tilde.shape}")

        W_gpu = W.to(device)

        # Original W stats
        W_energy_orig = (W**2).sum().item()
        S_w_orig = torch.linalg.svdvals(W.float().to(device))
        W_er_orig = _effective_rank(S_w_orig.cpu())
        print(
            f"\n  Original W: energy={W_energy_orig:.2e}, ER={W_er_orig:.1f}/{min(in_features, out_features)}"
        )

        # Per-band temporal analysis
        print(f"\n  Per-band temporal analysis:")
        print(
            f"  {'Band':>6} | {'Size':>5} | {'SeqFrac':>8} | {'X energy':>9} | {'Y energy':>9} "
            f"| {'X ER':>10} | {'Y ER':>10}"
        )
        print(f"  {'─' * 80}")

        x_total_energy = 0.0
        band_stats = []

        for b in seq_bands:
            X_band = X_tilde[:, b["start"] : b["end"], :]
            x_energy = np.sum(X_band.astype(np.float64) ** 2)
            x_total_energy += x_energy

            X_band_flat_np = X_band.reshape(-1, in_features)
            X_band_flat = torch.from_numpy(X_band_flat_np).float().to(device)
            Y_band_flat_np = (X_band_flat @ W_gpu.T).cpu().numpy()
            y_energy = np.sum(Y_band_flat_np.astype(np.float64) ** 2)

            er_x, er_y = analyze_band(
                X_band_flat_np,
                Y_band_flat_np,
                in_features,
                out_features,
            )

            band_stats.append(
                {
                    "label": b["label"],
                    "size": b["size"],
                    "x_energy": x_energy,
                    "y_energy": y_energy,
                    "er_x": er_x,
                    "er_y": er_y,
                }
            )
            del X_band_flat
            torch.cuda.empty_cache()

        total_y_energy = sum(bs["y_energy"] for bs in band_stats)

        for bs in band_stats:
            x_pct = 100 * bs["x_energy"] / x_total_energy
            y_pct = 100 * bs["y_energy"] / total_y_energy
            seq_pct = 100 * bs["size"] / seq_len
            print(
                f"  {bs['label']:>6} | {bs['size']:>5} | {seq_pct:7.2f}% | {x_pct:8.2f}% | {y_pct:8.2f}% "
                f"| {bs['er_x']:7.1f}/{in_features} | {bs['er_y']:7.1f}/{out_features}"
            )

        # MERA decomposition: per-band SVD rank analysis
        print(f"\n  MERA decomposition (feature-dim wavelet, level={FEAT_LEVEL})...")
        analyze_mera_decomposition(X_3d, W, wavelet=WAVELET, layer_name=layer_name)

        del X_3d, X_tilde
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
