"""Wavelet analysis of linear layer weights.

Combines spectral and real-space structure using multi-scale wavelets.
Tests different mother wavelets on covariance-reordered weights.
"""

import torch
import torch.nn as nn
import numpy as np
import pywt
import matplotlib.pyplot as plt
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
TARGET_SNR_DB = 35.0


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


def reorder_by_covariance(weight, input_activations):
    """Reorder weight matrix rows/cols by activation covariance importance.

    Returns:
        reordered_weight: Weight matrix with rows/cols reordered by covariance importance
        input_perm: Column permutation (input channels)
        output_perm: Row permutation (output channels)
    """
    W = weight.float().cpu()
    X = input_activations.float().cpu()

    out_features, in_features = W.shape

    # Input covariance
    cov = (X.T @ X) / X.shape[0]
    eigvals_in, eigvecs_in = torch.linalg.eigh(cov)
    eigvals_in = eigvals_in.flip(0)  # descending
    eigvecs_in = eigvecs_in.flip(1)

    # Reorder by descending eigenvalue energy
    input_perm = torch.argsort(eigvals_in, descending=True)

    # Output: use weight correlation as proxy (no output activations)
    col_norms = W.norm(dim=0)
    act_norms = torch.sqrt((X ** 2).mean(dim=0) + 1e-10)
    importance = col_norms * act_norms
    # Synthetic "output importance" from row norms
    row_norms = W.norm(dim=1)
    output_perm = torch.argsort(row_norms, descending=True)

    # Apply permutations
    W_reordered = W[output_perm, :][:, input_perm]

    return W_reordered, input_perm, output_perm


def wavelet_decompose_2d(weight, wavelet, level=None):
    """Perform 2D wavelet decomposition.

    Args:
        weight: (out_features, in_features) matrix
        wavelet: Wavelet name (e.g., 'db4', 'morlet', 'haar')
        level: Decomposition level (None = max possible)

    Returns:
        coeffs: Nested list structure from pywt.wavedec2
        slices: Dict mapping coefficient names to locations in flattened array
    """
    W = weight.numpy() if isinstance(weight, torch.Tensor) else weight

    if level is None:
        level = pywt.dwt_max_level(min(W.shape), wavelet)

    try:
        coeffs = pywt.wavedec2(W, wavelet, level=level)
    except Exception as e:
        print(f"  Wavelet '{wavelet}' failed: {e}")
        return None, None

    return coeffs, level


def coeffs_to_array(coeffs):
    """Flatten nested wavelet coefficient structure to 1D array for thresholding."""
    flat = []
    for item in coeffs:
        if isinstance(item, tuple):
            # Tuple of (cA, cH, cV, cD) at each level
            for sub in item:
                flat.append(sub.flatten())
        else:
            # Base level (cA)
            flat.append(item.flatten())
    return np.concatenate(flat)


def array_to_coeffs(flat_array, coeffs_template):
    """Reconstruct nested coefficient structure from flattened array."""
    result = []
    offset = 0

    for item in coeffs_template:
        if isinstance(item, tuple):
            # Tuple of (cA, cH, cV, cD)
            subs = []
            for sub in item:
                size = sub.size
                subs.append(flat_array[offset:offset+size].reshape(sub.shape))
                offset += size
            result.append(tuple(subs))
        else:
            # Base level
            size = item.size
            result.append(flat_array[offset:offset+size].reshape(item.shape))
            offset += size

    return result


def threshold_coefficients(coeffs, threshold):
    """Hard threshold wavelet coefficients."""
    thresholded = []
    for item in coeffs:
        if isinstance(item, tuple):
            subs = tuple(np.sign(s) * np.maximum(np.abs(s) - threshold, 0) for s in item)
            thresholded.append(subs)
        else:
            thresholded.append(np.sign(item) * np.maximum(np.abs(item) - threshold, 0))
    return thresholded


def reconstruct_from_wavelets(coeffs, wavelet):
    """Reconstruct weight matrix from wavelet coefficients."""
    return pywt.waverec2(coeffs, wavelet)


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


def wavelet_compression_sweep(weight, input_activations, wavelet, original_output,
                              original_params, target_snr_db, output_inv_perm=None, verbose=True):
    """Sweep threshold to find SNR vs. params tradeoff.

    Returns:
        results: Dict with threshold -> {snr, params, residual_er, wavelet_er, ...}
    """
    W = weight.float().cpu()
    X = input_activations.float().cpu()

    out_features, in_features = W.shape

    # Decompose
    coeffs, level = wavelet_decompose_2d(W, wavelet)
    if coeffs is None:
        return None

    # Flatten for thresholding
    flat_coeffs = coeffs_to_array(coeffs)

    # Compute ER of wavelet coefficients
    coeff_er = wavelet_er(flat_coeffs)

    # Sweep thresholds
    sorted_mags = np.sort(np.abs(flat_coeffs))
    sorted_mags = sorted_mags[sorted_mags > 0]

    results = {}
    thresholds = np.percentile(sorted_mags, [0, 1, 5, 10, 20, 30, 50, 70, 90])

    if verbose:
        print(f"    Wavelet '{wavelet}' (level={level}, coeff_ER={coeff_er:.0f}):")
        print(f"    {'threshold':>12s}  {'SNR':>8s}  {'noise':>8s}  {'params':>8s}  {'resid_ER':>8s}")

    for thresh in thresholds:
        # Threshold and reconstruct
        thresholded = threshold_coefficients(coeffs, float(thresh))
        W_recon = reconstruct_from_wavelets(thresholded, wavelet)

        if W_recon is None or W_recon.shape != W.shape:
            continue

        W_recon_t = torch.from_numpy(W_recon).float().cpu()

        # Count nonzero coefficients
        num_nonzero = np.count_nonzero(flat_coeffs[np.abs(flat_coeffs) > thresh])

        # Compute SNR on activations
        channel_scale = torch.sqrt((X ** 2).mean(dim=0) + 1e-10)

        with torch.no_grad():
            output_recon_raw = W_recon_t.float() @ X.float().T
            output_recon_raw = output_recon_raw.T.float()
            # Unpermute output features if permutation was applied
            if output_inv_perm is not None:
                output_recon = output_recon_raw[:, output_inv_perm]
            else:
                output_recon = output_recon_raw

        snr, sig_db, noise_db = _compute_snr(original_output, output_recon)

        # Residual ER (SVD-based, like ASVD)
        residual = W - W_recon_t
        residual_scaled = residual * channel_scale.unsqueeze(0)
        try:
            _, S, _ = torch.linalg.svd(residual_scaled, full_matrices=False)
            resid_er = _effective_rank(S)
        except:
            resid_er = 0.0

        results[float(thresh)] = {
            'snr': snr,
            'noise': noise_db,
            'params': num_nonzero,
            'params_pct': 100 * num_nonzero / (out_features * in_features),
            'resid_er': resid_er,
            'wavelet_er': coeff_er,
        }

        if verbose:
            print(f"    {thresh:12.4e}  {snr:8.2f}dB  {noise_db:8.1f}dB  "
                  f"{num_nonzero:8d}  {resid_er:8.0f}")

    return results


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

    # Get the layer
    print(f"\nCollecting activations for {LAYER_NAME}...")
    input_activations = collect_layer_activations(model, LAYER_NAME, dataloader)
    print(f"Collected {input_activations.shape[0]} samples")

    # Get weight
    parts = LAYER_NAME.split(".")
    layer = model
    for part in parts:
        layer = layer[int(part)] if part.isdigit() else getattr(layer, part)

    W = layer.weight.data.float().cpu()
    out_features, in_features = W.shape
    original_params = W.numel()

    # Original output
    with torch.no_grad():
        original_output = (W.float() @ input_activations.float().T).T.float()

    print(f"\n{'='*80}")
    print(f"Wavelet Analysis: {LAYER_NAME}")
    print(f"Shape: {W.shape}, Params: {original_params:,}")
    print(f"{'='*80}")

    # Reorder by covariance
    print(f"\nReordering by activation covariance...")
    W_reordered, input_perm, output_perm = reorder_by_covariance(W, input_activations)

    # IMPORTANT: permute activations the same way as weight columns
    # This preserves the computation: W_perm @ X_perm = W @ X (same output, reordered coordinates)
    X_reordered = input_activations[:, input_perm]

    # Compute baseline SNR (should be 0 dB = perfect reconstruction since we're just reordering)
    with torch.no_grad():
        output_reordered_raw = (W_reordered.float() @ X_reordered.float().T).T.float()
        # Unpermute the output features to restore original order
        # output_inv_perm tells us where each original feature ended up in permuted order
        # So we index [:, output_inv_perm] to reorder features
        output_inv_perm = torch.argsort(output_perm)
        output_reordered = output_reordered_raw[:, output_inv_perm]

    snr_baseline, sig_db, noise_db = _compute_snr(original_output, output_reordered)
    print(f"Reordering preserves output (SNR={snr_baseline:.2f}dB)")

    if abs(snr_baseline) > 0.1:
        print(f"  WARNING: SNR should be ~0dB, got {snr_baseline:.2f}dB")

    # Try wavelets
    wavelets = ['haar', 'db2', 'db4', 'db8', 'sym4']

    print(f"\n{'='*80}")
    print("Wavelet Compression Sweeps")
    print(f"{'='*80}")

    all_results = {}
    # Get the inverse output permutation to unpermute outputs in the sweep
    output_inv_perm = torch.argsort(output_perm)

    for wavelet in wavelets:
        print(f"\n  {wavelet}:")
        results = wavelet_compression_sweep(
            W_reordered, X_reordered, wavelet,
            original_output, original_params, TARGET_SNR_DB,
            output_inv_perm=output_inv_perm,
            verbose=True
        )
        if results:
            all_results[wavelet] = results

    # Compute effective ranks in different bases for comparison
    print(f"\n{'='*80}")
    print("Effective Rank Comparison Across Representations")
    print(f"{'='*80}")

    # Real-space SVD
    _, S_real, _ = torch.linalg.svd(W.float().cpu(), full_matrices=False)
    er_real = _effective_rank(S_real)
    print(f"\n  Real-space (standard SVD): ER = {er_real:.0f}/{min(out_features, in_features)}")
    for thresh in [0.90, 0.95, 0.99]:
        cumsum_e = torch.cumsum(S_real**2, dim=0) / (S_real**2).sum()
        n = int((cumsum_e < thresh).sum().item()) + 1
        print(f"    {thresh:.0%} energy: {n} modes")

    # Activation-weighted SVD (covariance)
    channel_scale = torch.sqrt((input_activations.float().cpu() ** 2).mean(dim=0) + 1e-10)
    W_cov = W.float().cpu() * channel_scale.unsqueeze(0)
    _, S_cov, _ = torch.linalg.svd(W_cov, full_matrices=False)
    er_cov = _effective_rank(S_cov)
    print(f"\n  Covariance-weighted SVD: ER = {er_cov:.0f}/{min(out_features, in_features)}")
    for thresh in [0.90, 0.95, 0.99]:
        cumsum_e = torch.cumsum(S_cov**2, dim=0) / (S_cov**2).sum()
        n = int((cumsum_e < thresh).sum().item()) + 1
        print(f"    {thresh:.0%} energy: {n} modes")

    # Wavelet coefficient ERs
    print(f"\n  Wavelet coefficient ERs (magnitude-based):")
    for wavelet, results in all_results.items():
        if results:
            # Get a representative threshold (at 50% params)
            target_params = 0.5 * original_params
            best_thresh = min(results.keys(),
                             key=lambda t: abs(results[t]['params'] - target_params))
            r = results[best_thresh]
            print(f"    {wavelet:8s}: wavelet_ER={r['wavelet_er']:6.0f}, "
                  f"SNR @ 50% = {r['snr']:6.2f}dB, residual_ER={r['resid_er']:.0f}")

    # Summary at 50% params
    print(f"\n{'='*80}")
    print("Summary: SNR at 50% params")
    print(f"{'='*80}")
    target_params = 0.5 * original_params
    print(f"\n  Real-space SVD (baseline): --")
    print(f"  Covariance SVD: --")
    for wavelet, results in all_results.items():
        # Find threshold closest to 50% params
        best_thresh = min(results.keys(),
                         key=lambda t: abs(results[t]['params'] - target_params))
        r = results[best_thresh]
        print(f"  {wavelet:8s}: SNR={r['snr']:6.2f}dB, params={r['params_pct']:6.1f}%, "
              f"residual_ER={r['resid_er']:.0f}")


if __name__ == "__main__":
    main()
