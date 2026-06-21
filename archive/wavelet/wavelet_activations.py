"""Wavelet analysis of input activations.

Tests whether activations have multi-scale structure along the sample/time dimension.
"""

import torch
import torch.nn as nn
import numpy as np
import pywt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Config
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "mit-han-lab/pile-val-backup"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048

LAYER_NAME = "model.layers.15.self_attn.q_proj"


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


def _compute_snr(original, approx):
    """Compute SNR in dB."""
    signal_power = torch.var(original)
    mse = torch.mean((original - approx) ** 2)
    snr_linear = signal_power / (mse + 1e-10)
    snr_db = 10 * torch.log10(snr_linear).item()
    signal_db = 10 * torch.log10(signal_power + 1e-10).item()
    noise_db = 10 * torch.log10(mse + 1e-10).item()
    return snr_db, signal_db, noise_db


def _effective_rank(singular_values):
    """Effective rank via Shannon entropy."""
    sv2 = singular_values.float() ** 2
    sv2 = sv2[sv2 > 0]
    if len(sv2) == 0:
        return 0.0
    p = sv2 / sv2.sum()
    H = -(p * p.log()).sum().item()
    return np.exp(H)


def wavelet_er(flat_coeffs):
    """Compute effective rank of wavelet coefficients (by magnitude)."""
    mags = np.abs(flat_coeffs)
    mags = mags[mags > 0]
    if len(mags) == 0:
        return 0.0
    mags_sq = mags ** 2
    p = mags_sq / mags_sq.sum()
    H = -(p * np.log(p + 1e-10)).sum()
    return np.exp(H)


def wavelet_decompose_1d(signal, wavelet, level=None):
    """Perform 1D wavelet decomposition.

    Args:
        signal: 1D array
        wavelet: Wavelet name
        level: Decomposition level

    Returns:
        coeffs: Nested list from pywt.wavedec
    """
    s = signal.numpy() if isinstance(signal, torch.Tensor) else signal

    if level is None:
        level = pywt.dwt_max_level(len(s), wavelet)

    try:
        coeffs = pywt.wavedec(s, wavelet, level=level)
    except Exception as e:
        print(f"  Wavelet '{wavelet}' failed: {e}")
        return None, None

    return coeffs, level


def coeffs_to_array_1d(coeffs):
    """Flatten nested 1D wavelet coefficient structure."""
    flat = []
    for item in coeffs:
        if isinstance(item, np.ndarray):
            flat.append(item.flatten())
        else:
            flat.append(np.asarray(item).flatten())
    return np.concatenate(flat)


def array_to_coeffs_1d(flat_array, coeffs_template):
    """Reconstruct nested 1D coefficient structure."""
    result = []
    offset = 0
    for item in coeffs_template:
        if isinstance(item, np.ndarray):
            size = item.size
            result.append(flat_array[offset:offset+size].reshape(item.shape))
            offset += size
    return result


def threshold_coefficients_1d(coeffs, threshold):
    """Hard threshold wavelet coefficients."""
    thresholded = []
    for item in coeffs:
        thresholded.append(np.sign(item) * np.maximum(np.abs(item) - threshold, 0))
    return thresholded


def reconstruct_from_wavelets_1d(coeffs, wavelet):
    """Reconstruct 1D signal from wavelet coefficients."""
    try:
        return pywt.waverec(coeffs, wavelet)
    except:
        return None


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

    # Get the layer and its weight
    print(f"\nCollecting activations for {LAYER_NAME}...")
    X = collect_layer_activations(model, LAYER_NAME, dataloader)
    print(f"Collected activations: shape {X.shape}")

    parts = LAYER_NAME.split(".")
    layer = model
    for part in parts:
        layer = layer[int(part)] if part.isdigit() else getattr(layer, part)

    W = layer.weight.data.float().cpu()
    out_features, in_features = W.shape

    # Original output
    with torch.no_grad():
        original_output = (W.float() @ X.float().T).T.float()

    print(f"\n{'='*80}")
    print(f"Wavelet Analysis on Activations: {LAYER_NAME}")
    print(f"Activation matrix: {X.shape} ({X.numel():,} values)")
    print(f"{'='*80}")

    # Compute baseline activation spectrum
    X_cpu = X.float().cpu()
    _, S_X, _ = torch.linalg.svd(X_cpu, full_matrices=False)
    er_X = _effective_rank(S_X)
    print(f"\nActivation matrix spectrum:")
    print(f"  Effective rank: {er_X:.0f}/{min(X.shape)}")
    for thresh in [0.90, 0.95, 0.99]:
        cumsum_e = torch.cumsum(S_X**2, dim=0) / (S_X**2).sum()
        n = int((cumsum_e < thresh).sum().item()) + 1
        print(f"  {thresh:.0%} energy: {n} modes")

    # Try wavelets on individual feature channels
    # For each feature (column) in X, apply 1D wavelet decomposition along sample dimension
    wavelets = ['haar', 'db2', 'db4', 'db8', 'sym4']

    print(f"\n{'='*80}")
    print("Wavelet Compression: Per-Feature 1D Decomposition")
    print(f"{'='*80}")

    all_results = {}

    for wavelet in wavelets:
        print(f"\n  {wavelet}:")

        # Decompose each feature independently
        all_coeffs = []
        all_levels = []
        all_flat_coeffs = []

        for feat_idx in range(in_features):
            signal = X[:, feat_idx]
            coeffs, level = wavelet_decompose_1d(signal, wavelet)
            if coeffs is None:
                continue
            all_coeffs.append(coeffs)
            all_levels.append(level)
            flat = coeffs_to_array_1d(coeffs)
            all_flat_coeffs.append(flat)

        if not all_flat_coeffs:
            continue

        # Concatenate all feature coefficients
        all_flat_coeffs_concat = np.concatenate(all_flat_coeffs)
        coeff_er = wavelet_er(all_flat_coeffs_concat)

        print(f"    Decomposed {in_features} features")
        print(f"    Coefficient ER: {coeff_er:.0f}")
        print(f"    {'threshold':>12s}  {'SNR':>8s}  {'noise':>8s}  {'params':>8s}")

        # Sweep thresholds
        sorted_mags = np.sort(np.abs(all_flat_coeffs_concat))
        sorted_mags = sorted_mags[sorted_mags > 0]
        thresholds = np.percentile(sorted_mags, [0, 1, 5, 10, 20, 30, 50, 70, 90])

        results = {}
        for thresh in thresholds:
            # Threshold all features
            X_recon_list = []
            num_nonzero = 0

            for feat_idx in range(in_features):
                coeffs = all_coeffs[feat_idx]
                thresholded = threshold_coefficients_1d(coeffs, float(thresh))
                signal_recon = reconstruct_from_wavelets_1d(thresholded, wavelet)

                if signal_recon is None:
                    signal_recon = np.zeros(len(X[:, feat_idx]))

                # Pad/trim to original length
                if len(signal_recon) > len(X):
                    signal_recon = signal_recon[:len(X)]
                elif len(signal_recon) < len(X):
                    signal_recon = np.pad(signal_recon, (0, len(X) - len(signal_recon)))

                X_recon_list.append(signal_recon)
                num_nonzero += np.count_nonzero(all_flat_coeffs[feat_idx] > thresh)

            X_recon = np.column_stack(X_recon_list)
            X_recon_t = torch.from_numpy(X_recon).float().cpu()

            # Compute SNR
            with torch.no_grad():
                output_recon = (W.float() @ X_recon_t.T).T.float()

            snr, sig_db, noise_db = _compute_snr(original_output, output_recon)
            params_pct = 100 * num_nonzero / all_flat_coeffs_concat.size

            results[float(thresh)] = {
                'snr': snr,
                'noise': noise_db,
                'params': num_nonzero,
                'params_pct': params_pct,
            }

            print(f"    {thresh:12.4e}  {snr:8.2f}dB  {noise_db:8.1f}dB  {params_pct:8.1f}%")

        all_results[wavelet] = results

    # Summary
    print(f"\n{'='*80}")
    print("Summary: SNR at 50% coefficients retained")
    print(f"{'='*80}")
    target_pct = 50.0
    for wavelet, results in all_results.items():
        best_thresh = min(results.keys(),
                         key=lambda t: abs(results[t]['params_pct'] - target_pct))
        r = results[best_thresh]
        print(f"  {wavelet:8s}: SNR={r['snr']:6.2f}dB, params={r['params_pct']:6.1f}%")


if __name__ == "__main__":
    main()
