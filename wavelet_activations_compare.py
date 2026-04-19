"""Compare 1D vs 2D wavelet decomposition on activations.

1D: Per-feature decomposition along sample dimension
2D: Full activation matrix decomposition respecting both dimensions
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


# ============================================================================
# 2D WAVELET FUNCTIONS
# ============================================================================

def wavelet_decompose_2d(matrix, wavelet, level=None):
    """2D wavelet decomposition."""
    M = matrix.numpy() if isinstance(matrix, torch.Tensor) else matrix

    if level is None:
        level = pywt.dwt_max_level(min(M.shape), wavelet)

    try:
        coeffs = pywt.wavedec2(M, wavelet, level=level)
    except Exception as e:
        print(f"  Wavelet '{wavelet}' failed: {e}")
        return None, None

    return coeffs, level


def coeffs_to_array_2d(coeffs):
    """Flatten 2D wavelet coefficients."""
    flat = []
    for item in coeffs:
        if isinstance(item, tuple):
            for sub in item:
                flat.append(sub.flatten())
        else:
            flat.append(item.flatten())
    return np.concatenate(flat)


def threshold_coefficients_2d(coeffs, threshold):
    """Hard threshold 2D wavelet coefficients."""
    thresholded = []
    for item in coeffs:
        if isinstance(item, tuple):
            subs = tuple(np.sign(s) * np.maximum(np.abs(s) - threshold, 0) for s in item)
            thresholded.append(subs)
        else:
            thresholded.append(np.sign(item) * np.maximum(np.abs(item) - threshold, 0))
    return thresholded


def reconstruct_from_wavelets_2d(coeffs, wavelet):
    """Reconstruct from 2D wavelet coefficients."""
    try:
        return pywt.waverec2(coeffs, wavelet)
    except:
        return None


# ============================================================================
# 1D WAVELET FUNCTIONS
# ============================================================================

def wavelet_decompose_1d(signal, wavelet, level=None):
    """1D wavelet decomposition."""
    s = signal.numpy() if isinstance(signal, torch.Tensor) else signal

    if level is None:
        level = pywt.dwt_max_level(len(s), wavelet)

    try:
        coeffs = pywt.wavedec(s, wavelet, level=level)
    except Exception as e:
        return None, None

    return coeffs, level


def coeffs_to_array_1d(coeffs):
    """Flatten 1D wavelet coefficients."""
    flat = []
    for item in coeffs:
        flat.append(np.asarray(item).flatten())
    return np.concatenate(flat)


def threshold_coefficients_1d(coeffs, threshold):
    """Hard threshold 1D coefficients."""
    return [np.sign(item) * np.maximum(np.abs(item) - threshold, 0) for item in coeffs]


def reconstruct_from_wavelets_1d(coeffs, wavelet):
    """Reconstruct from 1D coefficients."""
    try:
        return pywt.waverec(coeffs, wavelet)
    except:
        return None


def test_2d_wavelet(X, W, original_output, wavelet, verbose=True, device='cuda'):
    """Test 2D wavelet decomposition on full activation matrix."""
    X_cpu = X.float().cpu()

    coeffs, level = wavelet_decompose_2d(X_cpu, wavelet)
    if coeffs is None:
        return None

    flat_coeffs = coeffs_to_array_2d(coeffs)
    coeff_er = wavelet_er(flat_coeffs)

    sorted_mags = np.sort(np.abs(flat_coeffs))
    sorted_mags = sorted_mags[sorted_mags > 0]
    thresholds = np.percentile(sorted_mags, [0, 1, 5, 10, 20, 30, 50, 70, 90])

    if verbose:
        print(f"    2D Wavelet '{wavelet}' (level={level}, coeff_ER={coeff_er:.0f}):")
        print(f"    {'threshold':>12s}  {'SNR':>8s}  {'noise':>8s}  {'params':>8s}  {'resid_ER':>8s}")

    results = {}
    for thresh in thresholds:
        thresholded = threshold_coefficients_2d(coeffs, float(thresh))
        X_recon = reconstruct_from_wavelets_2d(thresholded, wavelet)

        if X_recon is None or X_recon.shape != X_cpu.shape:
            continue

        X_recon_t = torch.from_numpy(X_recon).float().to(device)
        W_device = W.to(device)
        original_output_device = original_output.to(device)

        # Compute SNR
        with torch.no_grad():
            output_recon = (W_device @ X_recon_t.T).T.float()

        snr, sig_db, noise_db = _compute_snr(original_output_device, output_recon)

        # Compute residual effective rank on GPU
        residual = original_output_device - output_recon
        try:
            _, S_res, _ = torch.linalg.svd(residual.float(), full_matrices=False)
            resid_er = _effective_rank(S_res)
        except:
            resid_er = 0.0

        num_nonzero = np.count_nonzero(flat_coeffs > thresh)
        params_pct = 100 * num_nonzero / flat_coeffs.size

        results[float(thresh)] = {'snr': snr, 'noise': noise_db, 'params_pct': params_pct, 'resid_er': resid_er}

        if verbose:
            print(f"    {thresh:12.4e}  {snr:8.2f}dB  {noise_db:8.1f}dB  {params_pct:8.1f}%  {resid_er:8.0f}")

    return results


def test_1d_wavelet(X, W, original_output, wavelet, verbose=True, device='cuda'):
    """Test 1D wavelet decomposition per-feature."""
    X_cpu = X.float().cpu()
    in_features = X_cpu.shape[1]
    num_samples = X_cpu.shape[0]

    all_coeffs = []
    all_levels = []

    # Batch decompose: process in chunks to reduce loop overhead
    print(f"      Decomposing {in_features} features...", end='', flush=True)
    for feat_idx in range(in_features):
        signal = X_cpu[:, feat_idx]
        coeffs, level = wavelet_decompose_1d(signal, wavelet)
        if coeffs is None:
            continue
        all_coeffs.append(coeffs)
        all_levels.append(level)
    print(" done")

    if not all_coeffs:
        return None

    # Concatenate all feature coefficients
    all_flat_coeffs = []
    for feat_idx in range(in_features):
        flat = coeffs_to_array_1d(all_coeffs[feat_idx])
        all_flat_coeffs.append(flat)
    all_flat_coeffs_concat = np.concatenate(all_flat_coeffs)
    coeff_er = wavelet_er(all_flat_coeffs_concat)

    sorted_mags = np.sort(np.abs(all_flat_coeffs_concat))
    sorted_mags = sorted_mags[sorted_mags > 0]
    thresholds = np.percentile(sorted_mags, [0, 1, 5, 10, 20, 30, 50, 70, 90])

    if verbose:
        print(f"    1D Wavelet '{wavelet}' (coeff_ER={coeff_er:.0f}):")
        print(f"    {'threshold':>12s}  {'SNR':>8s}  {'noise':>8s}  {'params':>8s}  {'resid_ER':>8s}")

    results = {}
    for thresh in thresholds:
        X_recon_list = []
        num_nonzero = 0

        # Vectorize reconstruction where possible
        for feat_idx in range(in_features):
            coeffs = all_coeffs[feat_idx]
            thresholded = threshold_coefficients_1d(coeffs, float(thresh))
            signal_recon = reconstruct_from_wavelets_1d(thresholded, wavelet)

            if signal_recon is None:
                signal_recon = np.zeros(num_samples)
            else:
                if len(signal_recon) > num_samples:
                    signal_recon = signal_recon[:num_samples]
                elif len(signal_recon) < num_samples:
                    signal_recon = np.pad(signal_recon, (0, num_samples - len(signal_recon)))

            X_recon_list.append(signal_recon)
            num_nonzero += np.count_nonzero(all_flat_coeffs[feat_idx] > thresh)

        X_recon = np.column_stack(X_recon_list)
        X_recon_t = torch.from_numpy(X_recon).float().to(device)
        W_device = W.to(device)
        original_output_device = original_output.to(device)

        # Compute SNR
        with torch.no_grad():
            output_recon = (W_device @ X_recon_t.T).T.float()

        snr, sig_db, noise_db = _compute_snr(original_output_device, output_recon)

        # Compute residual effective rank on GPU
        residual = original_output_device - output_recon
        try:
            _, S_res, _ = torch.linalg.svd(residual.float(), full_matrices=False)
            resid_er = _effective_rank(S_res)
        except:
            resid_er = 0.0

        params_pct = 100 * num_nonzero / all_flat_coeffs_concat.size

        results[float(thresh)] = {'snr': snr, 'noise': noise_db, 'params_pct': params_pct, 'resid_er': resid_er}

        if verbose:
            print(f"    {thresh:12.4e}  {snr:8.2f}dB  {noise_db:8.1f}dB  {params_pct:8.1f}%  {resid_er:8.0f}")

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

    # Get the layer and activations
    print(f"\nCollecting activations for {LAYER_NAME}...")
    X = collect_layer_activations(model, LAYER_NAME, dataloader)
    print(f"Collected activations: shape {X.shape}")

    parts = LAYER_NAME.split(".")
    layer = model
    for part in parts:
        layer = layer[int(part)] if part.isdigit() else getattr(layer, part)

    W = layer.weight.data.float().cpu()

    # Original output (using ACTIVATIONS, not weights)
    with torch.no_grad():
        original_output = (W.float() @ X.float().T).T.float()

    print(f"\n{'='*80}")
    print(f"Wavelet Analysis on ACTIVATIONS (NOT weights): {LAYER_NAME}")
    print(f"Activation matrix: {X.shape}")
    print(f"{'='*80}")

    wavelets = ['haar', 'db2', 'db4']

    all_1d_results = {}
    all_2d_results = {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    for wavelet in wavelets:
        print(f"\n  {wavelet}:")
        result_1d = test_1d_wavelet(X, W, original_output, wavelet, verbose=True, device=device)
        if result_1d:
            all_1d_results[wavelet] = result_1d

        print()
        result_2d = test_2d_wavelet(X, W, original_output, wavelet, verbose=True, device=device)
        if result_2d:
            all_2d_results[wavelet] = result_2d

    # Summary comparison
    print(f"\n{'='*80}")
    print("Summary: SNR at 50% coefficients retained")
    print(f"{'='*80}")
    target_pct = 50.0

    print(f"\n  1D (per-feature):")
    for wavelet, results in all_1d_results.items():
        best_thresh = min(results.keys(),
                         key=lambda t: abs(results[t]['params_pct'] - target_pct))
        r = results[best_thresh]
        print(f"    {wavelet:6s}: SNR={r['snr']:6.2f}dB, noise={r['noise']:6.1f}dB, "
              f"params={r['params_pct']:6.1f}%, resid_ER={r['resid_er']:6.0f}")

    print(f"\n  2D (full matrix):")
    for wavelet, results in all_2d_results.items():
        best_thresh = min(results.keys(),
                         key=lambda t: abs(results[t]['params_pct'] - target_pct))
        r = results[best_thresh]
        print(f"    {wavelet:6s}: SNR={r['snr']:6.2f}dB, noise={r['noise']:6.1f}dB, "
              f"params={r['params_pct']:6.1f}%, resid_ER={r['resid_er']:6.0f}")


if __name__ == "__main__":
    main()
