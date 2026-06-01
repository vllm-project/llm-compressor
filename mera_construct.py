"""Construct MERA isometries for X and Y activations.

For each wavelet band, compute the data covariance, eigendecompose, and build
an isometry V_k that projects the band down to the subspace retaining 99.9% energy.
Then verify end-to-end: DWT → project each band → unproject → IDWT → measure SNR.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
import numpy as np
import pywt
import sys
import gc

sys.path.insert(0, "/home/brian-dellabetta/projects/llm-compressor")
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
DWT_MODE = "periodization"
FEAT_LEVEL = 4
ENERGY_TARGET = 0.999


def forward_dwt(x_np, wavelet=WAVELET, level=FEAT_LEVEL):
    coeffs = pywt.wavedec(x_np, wavelet, level=level, mode=DWT_MODE, axis=1)
    band_sizes = [c.shape[1] for c in coeffs]
    return np.concatenate(coeffs, axis=1), band_sizes


def inverse_dwt(coeffs_concat, band_sizes, wavelet=WAVELET):
    coeffs = []
    offset = 0
    for bs in band_sizes:
        coeffs.append(coeffs_concat[:, offset:offset + bs])
        offset += bs
    return pywt.waverec(coeffs, wavelet, mode=DWT_MODE, axis=1)


def get_band_structure(d, wavelet=WAVELET, level=FEAT_LEVEL):
    coeffs = pywt.wavedec(np.zeros(d), wavelet, level=level, mode=DWT_MODE)
    bands, offset = [], 0
    for i, c in enumerate(coeffs):
        label = f"A{level}" if i == 0 else f"D{level - i + 1}"
        bands.append({"label": label, "start": offset, "end": offset + len(c), "size": len(c)})
        offset += len(c)
    return bands


def build_mera_isometries(data_wav, bands, energy_target, device="cuda"):
    """Build per-band isometries from data covariance.

    For each band k:
      1. Extract band data: (n_samples, band_size)
      2. Covariance: C_k = data_k^T @ data_k / n  (band_size × band_size)
      3. Eigendecompose on GPU: C_k = V Λ V^T
      4. Choose r_k: smallest r such that sum(top-r eigenvalues) >= energy_target * total
      5. Isometry V_k: (band_size, r_k) — the top-r_k eigenvectors

    Returns list of isometries and per-band info.
    """
    n_samples = data_wav.shape[0]
    isometries = []
    band_info = []

    for b in bands:
        band_data = data_wav[:, b["start"]:b["end"]]
        band_size = b["size"]

        cov = torch.zeros(band_size, band_size, dtype=torch.float32, device=device)
        chunk = 16384
        for s in range(0, n_samples, chunk):
            e = min(s + chunk, n_samples)
            bc = torch.from_numpy(band_data[s:e].astype(np.float32)).to(device)
            cov += bc.T @ bc
            del bc
        cov /= n_samples

        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.flip(0).clamp(min=0)
        eigenvectors = eigenvectors.flip(1)

        total_energy = eigenvalues.sum().item()
        if total_energy < 1e-30:
            r_k = 1
        else:
            cum = torch.cumsum(eigenvalues, dim=0)
            idx = (cum / total_energy >= energy_target).nonzero(as_tuple=True)[0]
            r_k = idx[0].item() + 1 if len(idx) > 0 else len(eigenvalues)

        V_k = eigenvectors[:, :r_k].cpu()
        retained = eigenvalues[:r_k].sum().item() / total_energy * 100 if total_energy > 0 else 100

        isometries.append(V_k)
        band_info.append({
            "label": b["label"],
            "band_size": band_size,
            "rank": r_k,
            "energy_retained": retained,
        })

        del cov, eigenvalues, eigenvectors
        torch.cuda.empty_cache()

    return isometries, band_info


def project_bands(data_wav, bands, isometries):
    """Project each band through its isometry: data_band @ V_k → (n, r_k)."""
    projected = []
    for b, V_k in zip(bands, isometries):
        band_data = data_wav[:, b["start"]:b["end"]]
        projected.append(band_data @ V_k.numpy())
    return projected


def reconstruct_bands(projected, bands, isometries, total_dim):
    """Unproject each band and reassemble: proj_k @ V_k^T → (n, band_size)."""
    n = projected[0].shape[0]
    reconstructed = np.zeros((n, total_dim), dtype=np.float32)
    for b, proj_k, V_k in zip(bands, projected, isometries):
        reconstructed[:, b["start"]:b["end"]] = proj_k @ V_k.numpy().T
    return reconstructed


def measure_snr(original, reconstructed):
    sig = np.var(original)
    noise = np.mean((original - reconstructed) ** 2)
    if noise < 1e-30:
        return 200.0
    return 10 * np.log10(sig / noise)


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
    def collate_fn(batch):
        return {"input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in batch])}
    dataloader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)

    print(f"\nCollecting activations and weights...")
    layer_data = {}
    for layer_name in LAYER_NAMES:
        print(f"  {layer_name}...", end="", flush=True)
        X_3d = collect_activations(model, layer_name, dataloader)
        X_2d = X_3d.reshape(-1, X_3d.shape[2]).numpy().astype(np.float32)
        layer_ref = model
        for part in layer_name.split("."):
            layer_ref = layer_ref[int(part)] if part.isdigit() else getattr(layer_ref, part)
        W = layer_ref.weight.data.float().cpu()
        layer_data[layer_name] = {"X_2d": X_2d, "W": W}
        del X_3d
        print(f" X:{X_2d.shape} W:{W.shape}")

    del model, tokenizer
    torch.cuda.empty_cache(); gc.collect()
    print(f"Model deleted.\n")

    for layer_name in LAYER_NAMES:
        ld = layer_data[layer_name]
        X_2d = ld["X_2d"]
        W = ld["W"]
        N, d_in = X_2d.shape
        d_out = W.shape[0]

        print(f"{'=' * 100}")
        print(f"  {layer_name}  ({d_out}×{d_in})")
        print(f"{'=' * 100}")

        print(f"  Computing Y...", end="", flush=True)
        W_gpu = W.to(device)
        Y_chunks = []
        with torch.no_grad():
            for s in range(0, N, 16384):
                e = min(s + 16384, N)
                Y_chunks.append((torch.from_numpy(X_2d[s:e]).to(device) @ W_gpu.T).cpu().numpy())
        Y_2d = np.concatenate(Y_chunks, axis=0)
        del W_gpu, Y_chunks; torch.cuda.empty_cache()
        print(f" done. Y:{Y_2d.shape}")

        print(f"  DWT(X)...", end="", flush=True)
        X_wav, in_band_sizes = forward_dwt(X_2d)
        print(f" DWT(Y)...", end="", flush=True)
        Y_wav, out_band_sizes = forward_dwt(Y_2d)
        print(f" done.")

        in_bands = get_band_structure(d_in)
        out_bands = get_band_structure(d_out)

        for label, wav_data, orig_data, bands, band_sizes, d in [
            ("X", X_wav, X_2d, in_bands, in_band_sizes, d_in),
            ("Y", Y_wav, Y_2d, out_bands, out_band_sizes, d_out),
        ]:
            print(f"\n  ── {label} MERA ({ENERGY_TARGET*100:.1f}% energy target) ──")

            isometries, info = build_mera_isometries(wav_data, bands, ENERGY_TARGET, device)

            projected = project_bands(wav_data, bands, isometries)
            reconstructed_wav = reconstruct_bands(projected, bands, isometries, d)
            reconstructed = inverse_dwt(reconstructed_wav, band_sizes)[:, :d]

            wav_snr = measure_snr(wav_data, reconstructed_wav)
            full_snr = measure_snr(orig_data, reconstructed.astype(np.float32))

            total_rank = sum(bi["rank"] for bi in info)
            total_params = sum(bi["band_size"] * bi["rank"] for bi in info)
            orig_dims = d

            print(f"    {'Band':>6} {'Size':>6} {'Rank':>6} {'Retained':>10} {'Isometry params':>16}")
            print(f"    {'─' * 52}")
            for bi in info:
                iso_params = bi["band_size"] * bi["rank"]
                print(f"    {bi['label']:>6} {bi['band_size']:>6} {bi['rank']:>6} "
                      f"{bi['energy_retained']:>9.3f}% {iso_params:>16,}")

            print(f"    {'─' * 52}")
            print(f"    {'Total':>6} {d:>6} {total_rank:>6} {'':>10} {total_params:>16,}")
            print(f"\n    Compressed dims: {total_rank}/{d} ({100*total_rank/d:.1f}%)")
            print(f"    Isometry params: {total_params:,}")
            print(f"    Wavelet-domain SNR: {wav_snr:.1f}dB")
            print(f"    Full reconstruction SNR: {full_snr:.1f}dB")

        print()
        del X_wav, Y_wav, Y_2d
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
