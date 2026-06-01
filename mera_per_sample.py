"""Per-sample MERA bond dimension analysis.

For each activation vector (single token's X̃ or Ỹ), measure the Schmidt rank
at each natural wavelet band cut. This tells us whether individual samples can
be represented as MERA states with low bond dimension.

Wavelet bands [A4(256) | D4(256) | D3(512) | D2(1024) | D1(2048)] define 4 cuts:
  Cut 1: {A4} vs {D4,D3,D2,D1} — reshape vector to (256, 3840), SVD
  Cut 2: {A4,D4} vs {D3,D2,D1} — reshape to (512, 3584), SVD
  Cut 3: {A4,D4,D3} vs {D2,D1} — reshape to (1024, 3072), SVD
  Cut 4: {A4,D4,D3,D2} vs {D1} — reshape to (2048, 2048), SVD

The Schmidt rank at each cut = the MERA bond dimension needed at that scale.
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
ENERGY_THRESHOLDS = [0.99, 0.999, 0.9999]
N_SAMPLES = 512


def forward_dwt(x_np, wavelet=WAVELET, level=FEAT_LEVEL):
    coeffs = pywt.wavedec(x_np, wavelet, level=level, mode=DWT_MODE, axis=1)
    return np.concatenate(coeffs, axis=1)


def get_band_structure(d, wavelet=WAVELET, level=FEAT_LEVEL):
    coeffs = pywt.wavedec(np.zeros(d), wavelet, level=level, mode=DWT_MODE)
    bands, offset = [], 0
    for i, c in enumerate(coeffs):
        label = f"A{level}" if i == 0 else f"D{level - i + 1}"
        bands.append({"label": label, "start": offset, "end": offset + len(c), "size": len(c)})
        offset += len(c)
    return bands


def per_sample_mera_bonds(vectors, bands, device="cuda"):
    """Compute MERA bond dimensions for each sample vector at wavelet band cuts.

    The natural MERA cuts are between wavelet bands:
      Cut 1: {A4} vs {D4, D3, D2, D1}
      Cut 2: {A4, D4} vs {D3, D2, D1}
      Cut 3: {A4, D4, D3} vs {D2, D1}
      Cut 4: {A4, D4, D3, D2} vs {D1}

    For each sample, reshape the vector at each cut into a (left_dim, right_dim)
    matrix and SVD to find the Schmidt rank.

    Returns: dict mapping threshold → array of shape (n_samples, n_cuts).
    """
    n_samples = vectors.shape[0]
    n_bands = len(bands)
    n_cuts = n_bands - 1

    cut_boundaries = []
    for cut in range(1, n_bands):
        left_dim = sum(b["size"] for b in bands[:cut])
        right_dim = sum(b["size"] for b in bands[cut:])
        cut_boundaries.append((left_dim, right_dim))

    results = {t: np.zeros((n_samples, n_cuts), dtype=np.int32) for t in ENERGY_THRESHOLDS}

    chunk_size = 64
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        batch = torch.from_numpy(vectors[start:end].astype(np.float32)).to(device)

        for ci, (left_dim, right_dim) in enumerate(cut_boundaries):
            mat = batch[:, :left_dim + right_dim].reshape(end - start, left_dim, right_dim)

            for si in range(end - start):
                S = torch.linalg.svdvals(mat[si])
                sv_sq = S**2
                total = sv_sq.sum().item()
                if total < 1e-30:
                    for t in ENERGY_THRESHOLDS:
                        results[t][start + si, ci] = 0
                    continue
                cum = torch.cumsum(sv_sq, dim=0)
                for t in ENERGY_THRESHOLDS:
                    target = t * total
                    idx = (cum >= target).nonzero(as_tuple=True)[0]
                    r = idx[0].item() + 1 if len(idx) > 0 else len(S)
                    results[t][start + si, ci] = r

    return results


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
        N_total, d_in = X_2d.shape
        d_out = W.shape[0]

        print(f"{'=' * 100}")
        print(f"  {layer_name}  ({d_out}×{d_in})")
        print(f"{'=' * 100}")

        np.random.seed(42)
        sample_idx = np.random.choice(N_total, size=min(N_SAMPLES, N_total), replace=False)

        X_sub = X_2d[sample_idx]
        print(f"  Computing Y for {len(sample_idx)} samples...", end="", flush=True)
        W_gpu = W.to(device)
        with torch.no_grad():
            Y_sub = (torch.from_numpy(X_sub).to(device) @ W_gpu.T).cpu().numpy()
        del W_gpu; torch.cuda.empty_cache()
        print(f" done.")

        print(f"  DWT on X...", end="", flush=True)
        X_wav = forward_dwt(X_sub)
        print(f" done.")
        print(f"  DWT on Y...", end="", flush=True)
        Y_wav = forward_dwt(Y_sub)
        print(f" done.")

        bands_in = get_band_structure(d_in)
        bands_out = get_band_structure(d_out)

        for label, vecs, bands in [
            ("X̃", X_wav, bands_in),
            ("Ỹ", Y_wav, bands_out),
        ]:
            n_cuts = len(bands) - 1
            band_info = " ".join(f"{b['label']}[{b['size']}]" for b in bands)
            print(f"\n  {label} per-sample MERA bonds ({len(sample_idx)} samples):")
            print(f"    Bands: {band_info}")

            bond_results = per_sample_mera_bonds(vecs, bands, device)

            cut_labels = []
            max_possible = []
            for ci in range(n_cuts):
                left = " + ".join(b["label"] for b in bands[:ci + 1])
                right = " + ".join(b["label"] for b in bands[ci + 1:])
                cut_labels.append(f"{{{left}}} | {{{right}}}")
                left_dim = sum(b["size"] for b in bands[:ci + 1])
                right_dim = sum(b["size"] for b in bands[ci + 1:])
                max_possible.append(min(left_dim, right_dim))

            for t in ENERGY_THRESHOLDS:
                bonds = bond_results[t]
                mean_bonds = bonds.mean(axis=0)
                std_bonds = bonds.std(axis=0)
                max_bonds = bonds.max(axis=0)

                t_label = f"{t*100:.1f}%" if t < 0.999 else f"{t*100:.2f}%"
                print(f"\n    Energy threshold: {t_label}")
                print(f"    {'Cut':>4} {'MaxPoss':>8} {'Mean χ':>8} {'Std':>8} {'Max χ':>8} {'Mean%':>7}  Partition")
                print(f"    {'─' * 80}")

                for ci in range(n_cuts):
                    mp = max_possible[ci]
                    mn = mean_bonds[ci]
                    sd = std_bonds[ci]
                    mx = max_bonds[ci]
                    ratio = mn / mp * 100 if mp > 0 else 0
                    print(f"    {ci+1:>4} {mp:>8} {mn:>8.1f} {sd:>8.1f} {mx:>8} {ratio:>6.1f}%  {cut_labels[ci]}")

                total_mean = mean_bonds.sum()
                total_max = sum(max_possible)
                print(f"    {'─' * 80}")
                print(f"    {'Sum':>4} {total_max:>8} {total_mean:>8.0f} {'':>8} {int(max_bonds.sum()):>8} "
                      f"{total_mean/total_max*100:>6.1f}%")

        print()
        del X_sub, Y_sub, X_wav, Y_wav
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
