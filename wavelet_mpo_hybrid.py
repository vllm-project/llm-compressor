"""Wavelets (99% energy) + MPO hybrid compression on activations."""

import torch
import numpy as np
import pywt
from scipy.fftpack import dct
import tensorly as tl
from tensorly.decomposition import tensor_train
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Import all the helper functions from wavelet_cascade
import sys
sys.path.insert(0, '/home/brian-dellabetta/projects/llm-compressor')
from wavelet_cascade import (
    get_calib_dataset, collect_layer_activations,
    _compute_snr, _effective_rank, _cka, _cosine_similarity_mean,
    _kl_divergence, _ssim, _residual_sparsity_wavelet,
    wavelet_decompose_2d, coeffs_to_array_2d, threshold_coefficients_2d,
    reconstruct_from_wavelets_2d
)

# Config
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "mit-han-lab/pile-val-backup"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048

LAYER_NAME = "model.layers.15.self_attn.q_proj"


def apply_2d_wavelet_energy_threshold(matrix, wavelet, energy_threshold=0.99, device='cuda'):
    """Apply 2D wavelet with energy threshold."""
    matrix_cpu = matrix.float().cpu()

    coeffs, level = wavelet_decompose_2d(matrix_cpu, wavelet)
    if coeffs is None:
        return None, 0.0

    flat_coeffs = coeffs_to_array_2d(coeffs)
    mags = np.abs(flat_coeffs)

    sorted_indices = np.argsort(-mags)
    sorted_mags = mags[sorted_indices]
    energy_cumsum = np.cumsum(sorted_mags ** 2)
    total_energy = energy_cumsum[-1]

    target_energy = energy_threshold * total_energy
    num_needed = np.searchsorted(energy_cumsum, target_energy) + 1
    threshold = sorted_mags[num_needed - 1] if num_needed < len(sorted_mags) else 0.0

    thresholded = threshold_coefficients_2d(coeffs, float(threshold))
    matrix_recon = reconstruct_from_wavelets_2d(thresholded, wavelet)

    if matrix_recon is None or matrix_recon.shape != matrix_cpu.shape:
        return None, 0.0

    energy_retained = energy_cumsum[num_needed - 1] / total_energy * 100.0
    return torch.from_numpy(matrix_recon).float().to(device), energy_retained


def _tt_on_wavelet_coefficients(matrix, wavelet, rank=None):
    """Apply TT decomposition in wavelet space.

    1. Decompose matrix to wavelets (sparse, ordered coefficients)
    2. Flatten in hierarchical order (coarse to fine)
    3. Apply 1D TT to wavelet coefficient vector
    4. Reshape and inverse transform back to matrix

    Args:
        matrix: 2D weight matrix
        wavelet: wavelet type
        rank: TT bond dimension

    Returns:
        Reconstructed matrix (after TT on wavelet coefficients)
    """
    matrix_torch = matrix if isinstance(matrix, torch.Tensor) else torch.from_numpy(matrix).float()
    matrix_cpu = matrix_torch.float().cpu()

    # Stage 1: Decompose to wavelets
    coeffs, level = wavelet_decompose_2d(matrix_cpu, wavelet)
    if coeffs is None:
        return None

    # Stage 2: Flatten coefficients in hierarchical order (coarse to fine)
    flat_coeffs = coeffs_to_array_2d(coeffs)
    coeffs_tensor = torch.from_numpy(flat_coeffs).float()

    # Stage 3: Apply 1D TT to wavelet coefficient vector
    try:
        if rank is None:
            rank = min(len(flat_coeffs) // 2, 1000)
        else:
            rank = min(rank, len(flat_coeffs))

        # Simple 1D TT: decompose as vector and apply low-rank
        # For 1D, use SVD on reshaped tensor
        # Reshape flat_coeffs to 2D, apply SVD, keep top rank components
        coeffs_np = flat_coeffs.reshape(-1, 1)  # (n_coeffs, 1)
        coeffs_reshaped = torch.from_numpy(flat_coeffs).float().reshape(-1, 1)

        # For 1D vector approximation, we can use a different strategy:
        # Sort by magnitude, keep top rank worth of energy
        mags = np.abs(flat_coeffs)
        sorted_indices = np.argsort(-mags)

        # Find rank that captures 99% of energy
        if rank is None:
            energies_sq = (mags ** 2)
            cumsum = np.cumsum(energies_sq[sorted_indices])
            total_energy = cumsum[-1]
            target_energy = 0.99 * total_energy
            rank = np.searchsorted(cumsum, target_energy) + 1

        # Create approximation by thresholding
        threshold = mags[sorted_indices[rank - 1]] if rank < len(mags) else 0
        flat_coeffs_approx = flat_coeffs.copy()
        flat_coeffs_approx[mags < threshold] = 0

    except Exception as e:
        print(f"Wavelet TT failed: {e}")
        return None

    # Stage 4: Inverse transform back to matrix space
    try:
        # Reconstruct coefficient structure from flattened array
        # This is a simplified version - just use the thresholded coefficients
        coeffs_reconstructed = []
        offset = 0

        for item in coeffs:
            if isinstance(item, tuple):
                subs = []
                for sub in item:
                    size = sub.size
                    sub_data = flat_coeffs_approx[offset:offset+size].reshape(sub.shape)
                    subs.append(sub_data)
                    offset += size
                coeffs_reconstructed.append(tuple(subs))
            else:
                size = item.size
                sub_data = flat_coeffs_approx[offset:offset+size].reshape(item.shape)
                coeffs_reconstructed.append(sub_data)
                offset += size

        # Inverse wavelet transform
        matrix_recon = reconstruct_from_wavelets_2d(coeffs_reconstructed, wavelet)
        if matrix_recon is None:
            return None

        return torch.from_numpy(matrix_recon).float()

    except Exception as e:
        print(f"Wavelet reconstruction failed: {e}")
        return None


def apply_wavelets_mpo_hybrid(X, W, original_output, wavelet, rank=None, device='cuda'):
    """Hybrid: wavelets on X and W, then low-rank SVD on W.

    Args:
        rank: Number of singular values to keep in W factorization. If None, keeps all.
    """
    original_output_device = original_output.to(device)

    # Stage 1: Wavelets on X (99% energy)
    X_recon, x_energy_retained = apply_2d_wavelet_energy_threshold(X, wavelet, energy_threshold=0.99, device=device)
    if X_recon is None:
        return None, None

    # Stage 2: Wavelets on W (99% energy)
    W_recon, w_energy_retained = apply_2d_wavelet_energy_threshold(W, wavelet, energy_threshold=0.99, device=device)
    if W_recon is None:
        return None, None

    # Stage 3: Low-rank SVD on sparse W_recon
    W_mpo = _tt_decompose_sparse_matrix(W_recon, rank=rank)
    if W_mpo is None:
        return None, None

    W_mpo_device = W_mpo.float().to(device)

    # Compute num params: (m x rank) + (rank) + (rank x n) = rank*(m+n+1)
    m, n = W_mpo.shape
    if rank is None:
        rank_used = min(m, n)
    else:
        rank_used = rank
    num_params_hybrid = rank_used * (m + n + 1)
    num_params_dense = m * n
    params_pct = 100.0 * num_params_hybrid / num_params_dense

    # Final output: W_mpo @ X_recon.T gives (4096, 67432), transpose to match original (67432, 4096)
    with torch.no_grad():
        final_output = (W_mpo_device @ X_recon.T).T.float()

    # Metrics
    snr, _ = _compute_snr(original_output_device, final_output)
    cka = _cka(original_output_device, final_output)
    kl = _kl_divergence(original_output_device, final_output)

    residual = original_output_device - final_output
    _, S_res, _ = torch.linalg.svd(residual.float(), full_matrices=False)
    er = _effective_rank(S_res)
    energy = (S_res ** 2).sum().item()

    _, S_baseline, _ = torch.linalg.svd(original_output_device.float(), full_matrices=False)
    energy_baseline = (S_baseline ** 2).sum().item()
    energy_pct = 100.0 * energy / energy_baseline

    sparsity_wav = _residual_sparsity_wavelet(residual, wavelet)

    return final_output, (snr, energy_pct, er, cka, kl, sparsity_wav, x_energy_retained, w_energy_retained, params_pct)


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

    print("Loading calibration dataset...")
    calib_dataset = get_calib_dataset(tokenizer)

    def collate_fn(batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id or 0
        )
        return {"input_ids": input_ids}

    dataloader = DataLoader(calib_dataset, batch_size=4, collate_fn=collate_fn)

    print(f"\nCollecting activations for {LAYER_NAME}...")
    X = collect_layer_activations(model, LAYER_NAME, dataloader)
    print(f"Collected activations: shape {X.shape}")

    parts = LAYER_NAME.split(".")
    layer = model
    for part in parts:
        layer = layer[int(part)] if part.isdigit() else getattr(layer, part)

    W = layer.weight.data.float().cpu()

    with torch.no_grad():
        original_output = (W.float() @ X.float().T).T.float()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    wavelets = ['haar', 'db2', 'db4']

    print(f"{'='*190}")
    print("Wavelets (99% energy on X & W) + Low-Rank SVD on Sparse W")
    print(f"{'='*190}\n")

    # Sweep over different ranks
    ranks_to_test = [10, 20, 30, 50, 100, 200, 500, 1000, None]

    for wavelet in wavelets:
        print(f"\n{wavelet}:")
        print(f"  {'Rank':>8s} {'Params %':>10s} {'SNR':>10s} {'ER Real':>10s} {'Energy %':>10s} {'CKA':>10s} {'KL Div':>10s}")
        print(f"  {'-'*85}")

        for rank in ranks_to_test:
            final_output, metrics = apply_wavelets_mpo_hybrid(X, W, original_output, wavelet, rank=rank, device=device)
            if metrics:
                snr, energy_pct, er, cka, kl, sparsity_wav, x_energy, w_energy, params_pct = metrics
                rank_label = f"{rank}" if rank is not None else "full"
                print(f"  {rank_label:>8s} {params_pct:10.1f}% {snr:10.2f}dB {er:10.0f} {energy_pct:9.1f}% {cka:10.4f} {kl:10.4f}")
            else:
                rank_label = f"{rank}" if rank is not None else "full"
                print(f"  {rank_label:>8s} Failed")


if __name__ == "__main__":
    main()
