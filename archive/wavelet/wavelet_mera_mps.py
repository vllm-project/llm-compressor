"""Reyes & Stoudenmire MERA+MPS for Linear Layer Compression.

Architecture from arXiv:2001.08286:
1. Feature map: each input x_i → φ(x_i) = [1, x_i] (rank-1 MPS)
2. MERA coarse-graining: DWT reduces features by 2× per level
3. Weight MPS: trainable tensor network on coarse-grained features, optimized via DMRG/ALS

The MPS is trained on calibration data (minimizing ||Y - MPS(X)||²), not by
decomposing W. This exploits the low entanglement in Y activations.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
import torch.nn as nn
import numpy as np
import pywt
import sys
import gc

sys.path.insert(0, "/home/brian-dellabetta/projects/llm-compressor")
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

# ── Config ──────────────────────────────────────────────────────────────────────

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "mit-han-lab/pile-val-backup"
NUM_CALIBRATION_SAMPLES = 64
MAX_SEQUENCE_LENGTH = 2048
LAYER_NAMES = [
    "model.layers.15.self_attn.q_proj",
    # "model.layers.15.self_attn.k_proj",
    # "model.layers.15.self_attn.v_proj",
    # "model.layers.15.self_attn.o_proj",
    # "model.layers.15.mlp.gate_proj",
    # "model.layers.15.mlp.up_proj",
    # "model.layers.15.mlp.down_proj",
]
WAVELET = "db2"
DWT_MODE = "periodization"
FEAT_LEVEL = 4
N_TRAIN = 8192
OUTPUT_ENERGY = 0.99
OUTPUT_RANK_MAX = 256
CHI_VALUES = [2, 4, 8, 16, 32]
N_EPOCHS = 200
LR = 1e-3
BATCH_SIZE = 1024


# ── DWT ─────────────────────────────────────────────────────────────────────────


def forward_dwt(x_np, wavelet=WAVELET, level=FEAT_LEVEL):
    coeffs = pywt.wavedec(x_np, wavelet, level=level, mode=DWT_MODE, axis=1)
    band_sizes = [c.shape[1] for c in coeffs]
    return np.concatenate(coeffs, axis=1), band_sizes


def get_band_structure(d, wavelet=WAVELET, level=FEAT_LEVEL):
    coeffs = pywt.wavedec(np.zeros(d), wavelet, level=level, mode=DWT_MODE)
    bands, offset = [], 0
    for i, c in enumerate(coeffs):
        label = f"A{level}" if i == 0 else f"D{level - i + 1}"
        bands.append(
            {"label": label, "start": offset, "end": offset + len(c), "size": len(c)}
        )
        offset += len(c)
    return bands


# ── MPS Model ───────────────────────────────────────────────────────────────────


class WeightMPS(nn.Module):
    """Trainable MPS for supervised regression.

    Architecture from Reyes & Stoudenmire (2020):
    - n_sites sites, each with physical dim 2 (feature map [1, x_i])
    - Output index r at the last core
    - Forward: φ(x_1) ⊗ ... ⊗ φ(x_N') contracted with weight MPS → ℝ^r
    """

    def __init__(self, n_sites, chi, output_dim, device="cuda"):
        super().__init__()
        self.n_sites = n_sites
        self.chi = chi
        self.output_dim = output_dim
        cores = []
        for k in range(n_sites):
            chi_l = 1 if k == 0 else chi
            chi_r = output_dim if k == n_sites - 1 else chi
            core = torch.zeros(chi_l, 2, chi_r, device=device)
            m = min(chi_l, chi_r)
            core[:m, 0, :m] = torch.eye(m, device=device)
            core[:, 1, :] = torch.randn(chi_l, chi_r, device=device) * 0.01
            cores.append(nn.Parameter(core))
        self.cores = nn.ParameterList(cores)

    def forward(self, x):
        """x: (batch, n_sites). Returns (batch, output_dim)."""
        phi_0 = torch.stack([torch.ones_like(x[:, 0]), x[:, 0]], dim=-1)
        v = torch.einsum("np, apb -> nb", phi_0, self.cores[0])

        for k in range(1, self.n_sites):
            phi_k = torch.stack([torch.ones_like(x[:, k]), x[:, k]], dim=-1)
            v = torch.einsum("na, np, apb -> nb", v, phi_k, self.cores[k])

        return v

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ── Training ────────────────────────────────────────────────────────────────────


def train_mps_model(
    X_train,
    Y_train,
    n_sites,
    chi,
    output_dim,
    n_epochs=N_EPOCHS,
    lr=LR,
    batch_size=BATCH_SIZE,
    device="cuda",
):
    """Train MPS via Adam on minibatches. Returns trained model and final train SNR."""
    model = WeightMPS(n_sites, chi, output_dim, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N = X_train.shape[0]
    y_var = torch.var(Y_train).item()

    best_snr = -np.inf
    best_state = None

    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0

        for start in range(0, N, batch_size):
            idx = perm[start : min(start + batch_size, N)]
            x_batch = X_train[idx]
            y_batch = Y_train[idx]

            y_pred = model(x_batch)
            loss = torch.mean((y_pred - y_batch) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        mse = total_loss / N
        snr = 10 * np.log10(y_var / (mse + 1e-30))

        if snr > best_snr:
            best_snr = snr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"        epoch {epoch+1:>4d}: SNR={snr:.1f}dB, loss={mse:.4e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_snr


# ── Evaluation ──────────────────────────────────────────────────────────────────


def measure_snr(y_true, y_approx):
    signal_power = torch.var(y_true)
    noise_power = torch.mean((y_true - y_approx) ** 2)
    return 10 * torch.log10(signal_power / (noise_power + 1e-30)).item()


# ── Data Collection ─────────────────────────────────────────────────────────────


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


# ── Main ────────────────────────────────────────────────────────────────────────


def main():
    device = "cuda"

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map="cuda", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading calibration dataset...")
    ds = load_dataset(DATASET_ID, split=f"validation[:{NUM_CALIBRATION_SAMPLES * 20}]")

    def tokenize_and_filter(example):
        ids = tokenizer.encode(example["text"].strip())
        if len(ids) >= MAX_SEQUENCE_LENGTH:
            return {"input_ids": ids[:MAX_SEQUENCE_LENGTH], "keep": True}
        return {"input_ids": ids, "keep": False}

    ds = ds.map(tokenize_and_filter, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["keep"]).remove_columns(["keep"])
    ds = ds.shuffle(seed=42).select(range(min(NUM_CALIBRATION_SAMPLES, len(ds))))
    print(f"  {len(ds)} samples, each {MAX_SEQUENCE_LENGTH} tokens")

    def collate_fn(batch):
        return {
            "input_ids": torch.stack(
                [torch.tensor(item["input_ids"]) for item in batch]
            )
        }

    dataloader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)

    print(f"\nCollecting all activations and weights...")
    layer_data = {}
    for layer_name in LAYER_NAMES:
        print(f"  {layer_name}...", end="", flush=True)
        X_3d = collect_activations(model, layer_name, dataloader)
        X_2d = X_3d.reshape(-1, X_3d.shape[2]).numpy().astype(np.float32)
        layer_ref = model
        for part in layer_name.split("."):
            layer_ref = (
                layer_ref[int(part)] if part.isdigit() else getattr(layer_ref, part)
            )
        W = layer_ref.weight.data.float().cpu()
        layer_data[layer_name] = {"X_2d": X_2d, "W": W}
        del X_3d
        print(f" X:{X_2d.shape} W:{W.shape}")

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Model deleted.")

    for layer_name in LAYER_NAMES:
        ld = layer_data[layer_name]
        X_2d = ld["X_2d"]
        W = ld["W"]
        N_total, d_in = X_2d.shape
        d_out = W.shape[0]

        print(f"\n{'=' * 100}")
        print(f"  Layer: {layer_name}  ({d_out}×{d_in})")
        print(f"{'=' * 100}")

        print(f"  Computing Y...", end="", flush=True)
        W_gpu = W.to(device)
        Y_chunks = []
        with torch.no_grad():
            for s in range(0, N_total, 16384):
                e = min(s + 16384, N_total)
                Y_chunks.append(
                    (torch.from_numpy(X_2d[s:e]).to(device) @ W_gpu.T).cpu()
                )
        Y_full = torch.cat(Y_chunks, dim=0)
        del W_gpu, Y_chunks
        torch.cuda.empty_cache()
        print(f" done. Y: {Y_full.shape}")

        print(f"  DWT on X...", end="", flush=True)
        X_wav, band_sizes = forward_dwt(X_2d)
        bands = get_band_structure(d_in)
        band_info = " ".join(f"{b['label']}[{b['size']}]" for b in bands)
        print(f" done. Bands: {band_info}")

        print(f"  Output SVD on GPU...", end="", flush=True)
        Y_mean = Y_full.mean(dim=0, keepdim=True)
        Y_centered = Y_full - Y_mean
        Y_sub = Y_centered[: min(N_total, 8192)].to(device)
        _, S_y, Vt_y = torch.linalg.svd(Y_sub, full_matrices=False)
        cum_energy = torch.cumsum(S_y**2, dim=0)
        total_energy = cum_energy[-1].item()
        r = (
            int(
                (cum_energy / total_energy >= OUTPUT_ENERGY)
                .nonzero(as_tuple=True)[0][0]
                .item()
            )
            + 1
        )
        r = min(r, OUTPUT_RANK_MAX)
        V_r = Vt_y[:r].T.cpu()
        energy_captured = cum_energy[r - 1].item() / total_energy * 100
        del Y_sub, S_y, Vt_y, cum_energy
        torch.cuda.empty_cache()
        Y_proj_full = (Y_centered @ V_r).numpy()
        V_r_np = V_r.numpy()
        Y_mean_np = Y_mean.numpy()
        print(f" r={r}, captures {energy_captured:.1f}% of Y variance")

        np.random.seed(42)
        train_idx = np.random.choice(N_total, size=min(N_TRAIN, N_total), replace=False)

        orig_params = d_in * d_out
        output_proj_params = r * d_out
        print(
            f"  Output projection: {output_proj_params:,} params (r={r} × d_out={d_out})"
        )

        levels_to_test = [4, 3, 2, 1]
        all_results = []

        for level in levels_to_test:
            n_coarse_bands = FEAT_LEVEL - level + 1
            actual_n = sum(band_sizes[:n_coarse_bands])
            band_labels = " ".join(b["label"] for b in bands[:n_coarse_bands])

            print(f"\n  --- Level {level}: {actual_n} sites ({band_labels}) ---")

            X_coarse_train = torch.from_numpy(
                X_wav[train_idx, :actual_n].astype(np.float32)
            ).to(device)
            Y_proj_train = torch.from_numpy(
                Y_proj_full[train_idx].astype(np.float32)
            ).to(device)

            x_std = X_coarse_train.std(dim=0, keepdim=True).clamp(min=1e-6)
            X_coarse_train_norm = X_coarse_train / x_std

            level_results = []

            for chi in CHI_VALUES:
                if chi > actual_n:
                    continue
                print(f"\n    χ={chi}:")
                mps_model, train_snr = train_mps_model(
                    X_coarse_train_norm,
                    Y_proj_train,
                    actual_n,
                    chi,
                    r,
                    n_epochs=N_EPOCHS,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    device=device,
                )

                with torch.no_grad():
                    X_coarse_full = torch.from_numpy(
                        X_wav[:, :actual_n].astype(np.float32)
                    ).to(device)
                    X_coarse_full_norm = X_coarse_full / x_std

                    y_proj_pred = []
                    for s in range(0, N_total, 8192):
                        e = min(s + 8192, N_total)
                        y_proj_pred.append(mps_model(X_coarse_full_norm[s:e]).cpu())
                    y_proj_pred = torch.cat(y_proj_pred, dim=0)

                Y_approx = (y_proj_pred.numpy() @ V_r_np.T) + Y_mean_np

                full_snr = measure_snr(
                    Y_full, torch.from_numpy(Y_approx.astype(np.float32))
                )
                mps_params = mps_model.param_count()
                total_params = mps_params + output_proj_params
                ratio = orig_params / total_params

                print(
                    f"      Full SNR={full_snr:.1f}dB, MPS params={mps_params:,}, "
                    f"total={total_params:,} ({100*total_params/orig_params:.2f}%), "
                    f"ratio={ratio:.1f}×"
                )

                level_results.append(
                    {
                        "level": level,
                        "sites": actual_n,
                        "chi": chi,
                        "mps_params": mps_params,
                        "total_params": total_params,
                        "total_pct": 100 * total_params / orig_params,
                        "train_snr": train_snr,
                        "full_snr": full_snr,
                        "ratio": ratio,
                    }
                )

                del mps_model
                torch.cuda.empty_cache()

            all_results.extend(level_results)
            del X_coarse_train, Y_proj_train
            torch.cuda.empty_cache()

        print(f"\n  {'='*90}")
        print(f"  SUMMARY: {layer_name}")
        print(f"  {'='*90}")
        print(f"  Original params: {orig_params:,}")
        hdr = f"  {'Level':>5} {'Sites':>6} {'χ':>4} {'MPS Params':>12} {'Total':>12} {'%':>7} {'Train':>8} {'Full':>8} {'Ratio':>7}"
        print(hdr)
        print(f"  {'─' * 88}")
        for r_item in all_results:
            print(
                f"  {r_item['level']:>5} {r_item['sites']:>6} {r_item['chi']:>4} "
                f"{r_item['mps_params']:>12,} {r_item['total_params']:>12,} "
                f"{r_item['total_pct']:>6.2f}% {r_item['train_snr']:>7.1f}dB "
                f"{r_item['full_snr']:>7.1f}dB {r_item['ratio']:>6.1f}×"
            )

        del Y_full, X_wav, Y_proj_full, Y_np, Y_centered
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
