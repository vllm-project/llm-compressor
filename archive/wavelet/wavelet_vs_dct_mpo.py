"""Compare two MPO approaches: wavelets + wavelet-ordered TT vs DCT + frequency-ordered TT."""

import torch
import numpy as np
import pywt
from scipy.fftpack import dct, idct
import sys
sys.path.insert(0, '/home/brian-dellabetta/projects/llm-compressor')
from wavelet_cascade import (
    get_calib_dataset, collect_layer_activations,
    _compute_snr, _effective_rank, _cka, _kl_divergence,
    _residual_sparsity_wavelet,
    wavelet_decompose_2d, coeffs_to_array_2d, threshold_coefficients_2d,
    reconstruct_from_wavelets_2d
)
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


def approach_1_wavelet_lowrank(X, W, original_output, wavelet, rank=None, device='cuda'):
    """Approach 1: Wavelets on X (99% energy) + Low-rank SVD on original W (no thresholding)."""
    original_output_device = original_output.to(device)

    # X: wavelets (99% energy)
    X_recon, x_energy = apply_2d_wavelet_energy_threshold(X, wavelet, energy_threshold=0.99, device=device)
    if X_recon is None:
        return None, None

    # W: SVD low-rank on ORIGINAL dense W (no wavelet thresholding)
    W_torch = W.float().cpu()

    try:
        U, S, Vh = torch.linalg.svd(W_torch, full_matrices=False)

        if rank is None:
            rank_used = len(S)
        else:
            rank_used = min(rank, len(S))

        U_trunc = U[:, :rank_used]
        S_trunc = S[:rank_used]
        Vh_trunc = Vh[:rank_used, :]

        W_lowrank = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
    except Exception as e:
        print(f"  Lowrank failed: {e}")
        return None, None

    W_lowrank_device = W_lowrank.float().to(device)

    # Compute output
    with torch.no_grad():
        final_output = (W_lowrank_device @ X_recon.T).T.float()

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

    # Params: rank * (m + n + 1)
    m, n = W_lowrank.shape
    num_params = rank_used * (m + n + 1)
    params_pct = 100.0 * num_params / (m * n)

    return (snr, er, energy_pct, cka, kl, params_pct), None


def approach_2_dct_lowrank(X, W, original_output, wavelet, rank=None, device='cuda'):
    """Approach 2: Wavelets on X (99% energy) + Low-rank SVD on DCT of W (no thresholding)."""
    original_output_device = original_output.to(device)

    # X: wavelets (99% energy)
    X_recon, x_energy = apply_2d_wavelet_energy_threshold(X, wavelet, energy_threshold=0.99, device=device)
    if X_recon is None:
        return None, None

    # W: DCT decomposition of ORIGINAL W (no thresholding)
    W_np = W.float().cpu().numpy()
    W_dct = dct(dct(W_np, axis=0, norm='ortho'), axis=1, norm='ortho')

    # SVD low-rank on DCT coefficients
    W_dct_torch = torch.from_numpy(W_dct).float()

    try:
        U, S, Vh = torch.linalg.svd(W_dct_torch, full_matrices=False)

        if rank is None:
            rank_used = len(S)
        else:
            rank_used = min(rank, len(S))

        U_trunc = U[:, :rank_used]
        S_trunc = S[:rank_used]
        Vh_trunc = Vh[:rank_used, :]

        W_dct_lowrank = U_trunc @ torch.diag(S_trunc) @ Vh_trunc

        # Inverse DCT back to spatial domain
        W_dct_lowrank_np = W_dct_lowrank.float().cpu().numpy()
        W_spatial = idct(idct(W_dct_lowrank_np, axis=0, norm='ortho'), axis=1, norm='ortho')
        W_lowrank = torch.from_numpy(W_spatial).float()
    except Exception as e:
        print(f"  DCT+Lowrank failed: {e}")
        return None, None

    W_lowrank_device = W_lowrank.float().to(device)

    # Compute output
    with torch.no_grad():
        final_output = (W_lowrank_device @ X_recon.T).T.float()

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

    # Params: rank * (m + n + 1)
    m, n = W_lowrank.shape
    num_params = rank_used * (m + n + 1)
    params_pct = 100.0 * num_params / (m * n)

    return (snr, er, energy_pct, cka, kl, params_pct), None


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

    wavelet = 'db2'
    ranks_to_test = [10, 20, 30, 50, 100, 200, 500, 1000, None]

    print(f"{'='*140}")
    print(f"Approach 1: Wavelets on X (99%) + Low-Rank SVD on Dense W ({wavelet})")
    print(f"{'='*140}")
    print(f"{'Rank':>8s} {'Params %':>10s} {'SNR':>10s} {'ER Real':>10s} {'Energy %':>10s} {'CKA':>10s} {'KL Div':>10s}")
    print(f"{'-'*90}")

    for rank in ranks_to_test:
        metrics, _ = approach_1_wavelet_lowrank(X, W, original_output, wavelet, rank=rank, device=device)
        if metrics:
            snr, er, energy_pct, cka, kl, params_pct = metrics
            rank_label = f"{rank}" if rank is not None else "full"
            print(f"{rank_label:>8s} {params_pct:10.1f}% {snr:10.2f}dB {er:10.0f} {energy_pct:9.1f}% {cka:10.4f} {kl:10.4f}")
        else:
            rank_label = f"{rank}" if rank is not None else "full"
            print(f"{rank_label:>8s} Failed")

    print(f"\n{'='*140}")
    print(f"Approach 2: Wavelets on X (99%) + Low-Rank SVD on DCT of W ({wavelet})")
    print(f"{'='*140}")
    print(f"{'Rank':>8s} {'Params %':>10s} {'SNR':>10s} {'ER Real':>10s} {'Energy %':>10s} {'CKA':>10s} {'KL Div':>10s}")
    print(f"{'-'*90}")

    for rank in ranks_to_test:
        metrics, _ = approach_2_dct_lowrank(X, W, original_output, wavelet, rank=rank, device=device)
        if metrics:
            snr, er, energy_pct, cka, kl, params_pct = metrics
            rank_label = f"{rank}" if rank is not None else "full"
            print(f"{rank_label:>8s} {params_pct:10.1f}% {snr:10.2f}dB {er:10.0f} {energy_pct:9.1f}% {cka:10.4f} {kl:10.4f}")
        else:
            rank_label = f"{rank}" if rank is not None else "full"
            print(f"{rank_label:>8s} Failed")


if __name__ == "__main__":
    main()
