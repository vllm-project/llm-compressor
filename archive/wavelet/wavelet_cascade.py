"""Test cascaded wavelet approach: 2D + 1D on residual.

1. Apply 2D wavelets at 25% params to activations
2. Compute residual
3. Apply 1D wavelets at 25% params to residual
4. Compare combined SNR to single approaches
"""

import torch
import numpy as np
import pywt
from scipy.fftpack import dct
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import tensorly as tl
from tensorly.decomposition import tensor_train
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Config
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "mit-han-lab/pile-val-backup"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 4096

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
    noise_db = 10 * torch.log10(mse + 1e-10).item()
    return snr_db, noise_db


def _effective_rank(singular_values):
    """Effective rank via Shannon entropy."""
    sv2 = singular_values.float() ** 2
    sv2 = sv2[sv2 > 0]
    if len(sv2) == 0:
        return 0.0
    p = sv2 / sv2.sum()
    H = -(p * p.log()).sum().item()
    return np.exp(H)


def _residual_sparsity_wavelet(residual, wavelet="db4", energy_threshold=0.99):
    """Compute sparsity of residual in wavelet coefficient basis.

    Applies 2D wavelet decomposition to the residual matrix and measures:
    "What percentage of coefficients (sorted by magnitude) do we need to
    retain to capture energy_threshold of the total energy?"

    Returns:
        sparsity (0-100): percentage of coefficients needed
        Low sparsity = highly compressible in wavelet space
        High sparsity = diffuse/spread out in wavelet space
    """
    residual_cpu = residual.float().cpu()

    coeffs, _ = wavelet_decompose_2d(residual_cpu, wavelet)
    if coeffs is None:
        return 100.0

    flat_coeffs = coeffs_to_array_2d(coeffs)
    mags = np.abs(flat_coeffs)

    if len(mags) == 0 or mags.sum() == 0:
        return 100.0

    # Sort by magnitude descending
    sorted_mags = np.sort(mags)[::-1]
    energy_cumsum = np.cumsum(sorted_mags)
    total_energy = energy_cumsum[-1]

    # Find how many coefficients needed for threshold
    target_energy = energy_threshold * total_energy
    num_needed = np.searchsorted(energy_cumsum, target_energy) + 1
    sparsity_pct = 100.0 * num_needed / len(mags)

    return sparsity_pct


def _cka(X1, X2, sample_size=5000):
    """Centered Kernel Alignment between two activation matrices.

    CKA is invariant to scaling and orthogonal transformations.
    Range: [0, 1], where 1 = identical representations.

    Uses sampling to avoid OOM on large matrices.
    """
    # Sample to avoid OOM on large activation matrices
    n = X1.shape[0]
    if n > sample_size:
        # Random sample without replacement
        indices = torch.randperm(n, device=X1.device)[:sample_size]
        X1_sample = X1[indices]
        X2_sample = X2[indices]
    else:
        X1_sample = X1
        X2_sample = X2

    # Compute linear kernel matrices (Gram matrices)
    K1 = X1_sample @ X1_sample.T  # (sample_size, sample_size)
    K2 = X2_sample @ X2_sample.T  # (sample_size, sample_size)

    # Center the kernels
    n_sample = K1.shape[0]
    H = torch.eye(n_sample, device=K1.device) - (1.0 / n_sample)
    K1_centered = H @ K1 @ H
    K2_centered = H @ K2 @ H

    # CKA = <K1, K2>_F / (||K1||_F * ||K2||_F)
    numerator = (K1_centered * K2_centered).sum()
    denominator = torch.sqrt((K1_centered**2).sum() * (K2_centered**2).sum())

    if denominator < 1e-10:
        return 0.0
    return (numerator / denominator).item()


def _cosine_similarity_mean(X1, X2, sample_size=5000):
    """Mean cosine similarity between corresponding rows/samples.

    Range: [-1, 1], where 1 = identical directions.
    """
    # Sample to avoid memory issues
    n = X1.shape[0]
    if n > sample_size:
        indices = torch.randperm(n, device=X1.device)[:sample_size]
        X1_sample = X1[indices]
        X2_sample = X2[indices]
    else:
        X1_sample = X1
        X2_sample = X2

    # Normalize each sample to unit length
    X1_norm = X1_sample / (torch.norm(X1_sample, dim=1, keepdim=True) + 1e-10)
    X2_norm = X2_sample / (torch.norm(X2_sample, dim=1, keepdim=True) + 1e-10)

    # Cosine similarity per sample
    cos_sim_per_sample = (X1_norm * X2_norm).sum(dim=1)

    return cos_sim_per_sample.mean().item()


def _fisher_information_preservation(
    original_activations, approx_activations, sample_size=1000
):
    """Compute Fisher Information preservation via gradient flow analysis.

    Creates a dummy downstream task (linear layer + MSE) and compares
    how gradients flow backward through original vs. approximated activations.

    Returns:
        fisher_ratio: Ratio of gradient magnitudes (ideally 1.0)
        gradient_alignment: Cosine similarity of gradient directions (ideally 1.0)

    High values indicate the approximation preserves the information flow
    that the model would use for learning.
    """
    # Sample to avoid memory issues
    n = original_activations.shape[0]
    if n > sample_size:
        indices = torch.randperm(n, device=original_activations.device)[:sample_size]
        X_orig = original_activations[indices].clone().detach()
        X_approx = approx_activations[indices].clone().detach()
    else:
        X_orig = original_activations.clone().detach()
        X_approx = approx_activations.clone().detach()

    # Dummy downstream task: linear layer predicting random targets
    out_dim = min(128, X_orig.shape[1])
    W_dummy = torch.randn(
        out_dim,
        X_orig.shape[1],
        device=X_orig.device,
        dtype=X_orig.dtype,
        requires_grad=False,
    )
    # Use actual outputs as targets (so gradients are meaningful)
    target_y = (W_dummy @ X_orig.T).T.detach()

    # Compute gradients w.r.t. original activations
    X_orig.requires_grad_(True)
    y_orig = (W_dummy @ X_orig.T).T
    loss_orig = ((y_orig - target_y) ** 2).mean()
    loss_orig.backward()
    grad_orig = X_orig.grad.detach()

    # Compute gradients w.r.t. approximated activations
    X_approx.requires_grad_(True)
    y_approx = (W_dummy @ X_approx.T).T
    loss_approx = ((y_approx - target_y) ** 2).mean()
    loss_approx.backward()
    grad_approx = X_approx.grad.detach()

    # Fisher Information: ratio of gradient magnitudes
    fisher_orig_norm = torch.sqrt((grad_orig**2).sum())
    fisher_approx_norm = torch.sqrt((grad_approx**2).sum())
    fisher_ratio = fisher_approx_norm / (fisher_orig_norm + 1e-10)

    # Gradient alignment: cosine similarity of flattened gradients
    grad_orig_flat = grad_orig.reshape(-1)
    grad_approx_flat = grad_approx.reshape(-1)
    grad_alignment = torch.nn.functional.cosine_similarity(
        grad_orig_flat.unsqueeze(0), grad_approx_flat.unsqueeze(0), dim=1
    )

    return fisher_ratio.item(), grad_alignment.item()


def _kl_divergence(output1, output2, sample_size=5000):
    """KL divergence between output probability distributions.

    Assumes outputs are logits. Computes softmax and then KL(P||Q).
    """
    # Sample to avoid memory issues
    n = output1.shape[0]
    if n > sample_size:
        indices = torch.randperm(n, device=output1.device)[:sample_size]
        output1_sample = output1[indices]
        output2_sample = output2[indices]
    else:
        output1_sample = output1
        output2_sample = output2

    # Convert to probabilities via softmax
    p = torch.softmax(output1_sample, dim=-1)
    q = torch.softmax(output2_sample, dim=-1)

    # Clamp to avoid log(0)
    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)

    # KL divergence: sum over last dimension (across features), mean over samples
    kl = (p * (torch.log(p) - torch.log(q))).sum(dim=-1).mean()

    return kl.item()


def _ssim(X1, X2, sample_size=1000):
    """Mean Structural Similarity Index via correlation structure.

    SSIM measures pattern preservation. Range: [0, 1], where 1 = identical.
    Uses simplified computation: correlation between flattened vectors.

    Args:
        X1, X2: (n_samples, n_features) activation matrices
        sample_size: Number of samples to use (to avoid memory issues)

    Returns:
        SSIM value (correlation-based approximation)
    """
    # Sample to avoid memory issues
    n = X1.shape[0]
    if n > sample_size:
        indices = torch.randperm(n, device=X1.device)[:sample_size]
        X1_sample = X1[indices]
        X2_sample = X2[indices]
    else:
        X1_sample = X1
        X2_sample = X2

    X1_sample = X1_sample.float()
    X2_sample = X2_sample.float()

    # Flatten for correlation computation
    x1_flat = X1_sample.reshape(-1)
    x2_flat = X2_sample.reshape(-1)

    # Compute correlation between flattened vectors
    # Normalize to mean 0, std 1
    x1_norm = (x1_flat - x1_flat.mean()) / (x1_flat.std() + 1e-10)
    x2_norm = (x2_flat - x2_flat.mean()) / (x2_flat.std() + 1e-10)

    # Correlation is the cosine similarity of normalized vectors
    correlation = (x1_norm * x2_norm).mean()

    # Map to [0, 1] range
    ssim = (correlation + 1.0) / 2.0
    return ssim.clamp(min=0.0, max=1.0).item()


def _tt_reconstruct(cores):
    """Reconstruct matrix from TT cores.

    Args:
        cores: List of TT cores [C0, C1, ..., C_{d-1}] where each core is (r_left, n_i, r_right)

    Returns:
        Reconstructed matrix of shape (n0*n1*..., n_{d-1})
    """
    if len(cores) == 0:
        return None

    # Start with first core: (1, n0, r0)
    result = cores[0].squeeze(0)  # (n0, r0)

    # Contract with remaining cores
    for core in cores[1:]:
        # result: (current_n, r_in)
        # core: (r_in, n_i, r_out)
        result = torch.einsum("cr,cni->nri", result, core)
        result = result.reshape(result.shape[0] * result.shape[1], result.shape[2])

    return result


def _mpo_decompose_matrix(matrix, rank_list=None):
    """Decompose a 2D matrix using Tensor Train (MPO) decomposition.

    For a 2D matrix (n_samples, n_features), reshapes to 3D tensor then applies TT.
    This allows capturing structure across both dimensions.

    Args:
        matrix: (n_samples, n_features) torch tensor
        rank_list: TT ranks, if None uses automatic

    Returns:
        cores: List of TT cores
        original_shape: tuple (n_samples, n_features)
    """
    matrix_np = matrix.float().cpu().numpy()
    n_samples, n_features = matrix.shape

    # Reshape as 2D tensor for TT
    tensor = matrix_np.reshape(n_samples, n_features)

    # For a 2D tensor, TT needs 3 ranks: [r0, r1, r2]
    # r0=1 (left boundary), r1=bond dimension, r2=1 (right boundary)
    if rank_list is None:
        rank_list = [1, min(30, n_samples // 2, n_features // 2), 1]

    try:
        # Use tensorly's TT decomposition with correct rank specification
        # For 2D tensor, pass all 3 ranks
        tt_decomp = tensor_train(tensor, rank=rank_list)
        cores = [torch.from_numpy(c).float() for c in tt_decomp.factors]
        return cores, (n_samples, n_features)
    except Exception as e:
        print(f"TT decomposition failed: {e}")
        return None, None


def wavelet_er(flat_coeffs):
    """Compute effective rank of wavelet coefficients."""
    mags = np.abs(flat_coeffs)
    mags = mags[mags > 0]
    if len(mags) == 0:
        return 0.0
    mags_sq = mags**2
    p = mags_sq / mags_sq.sum()
    H = -(p * np.log(p + 1e-10)).sum()
    return np.exp(H)


# ============================================================================
# 2D FUNCTIONS
# ============================================================================


def wavelet_decompose_2d(matrix, wavelet):
    M = matrix.numpy() if isinstance(matrix, torch.Tensor) else matrix
    level = pywt.dwt_max_level(min(M.shape), wavelet)
    try:
        coeffs = pywt.wavedec2(M, wavelet, level=level)
    except:
        return None, None
    return coeffs, level


def coeffs_to_array_2d(coeffs):
    flat = []
    for item in coeffs:
        if isinstance(item, tuple):
            for sub in item:
                flat.append(sub.flatten())
        else:
            flat.append(item.flatten())
    return np.concatenate(flat)


def threshold_coefficients_2d(coeffs, threshold):
    thresholded = []
    for item in coeffs:
        if isinstance(item, tuple):
            subs = tuple(
                np.sign(s) * np.maximum(np.abs(s) - threshold, 0) for s in item
            )
            thresholded.append(subs)
        else:
            thresholded.append(np.sign(item) * np.maximum(np.abs(item) - threshold, 0))
    return thresholded


def reconstruct_from_wavelets_2d(coeffs, wavelet):
    try:
        return pywt.waverec2(coeffs, wavelet)
    except:
        return None


# ============================================================================
# 1D FUNCTIONS
# ============================================================================


def wavelet_decompose_1d(signal, wavelet):
    s = signal.numpy() if isinstance(signal, torch.Tensor) else signal
    level = pywt.dwt_max_level(len(s), wavelet)
    try:
        coeffs = pywt.wavedec(s, wavelet, level=level)
    except:
        return None, None
    return coeffs, level


def coeffs_to_array_1d(coeffs):
    flat = []
    for item in coeffs:
        flat.append(np.asarray(item).flatten())
    return np.concatenate(flat)


def threshold_coefficients_1d(coeffs, threshold):
    return [np.sign(item) * np.maximum(np.abs(item) - threshold, 0) for item in coeffs]


def reconstruct_from_wavelets_1d(coeffs, wavelet):
    try:
        return pywt.waverec(coeffs, wavelet)
    except:
        return None


def apply_2d_wavelet(X, W, original_output, wavelet, target_pct=25.0, device="cuda"):
    """Apply 2D wavelet at target_pct param retention.

    Returns:
        X_recon_t: Reconstructed activations from 2D wavelet
        residual_X: Residual in ACTIVATION space (X - X_recon)
        info: (snr, noise, residual_er)
    """
    X_cpu = X.float().cpu()

    coeffs, level = wavelet_decompose_2d(X_cpu, wavelet)
    if coeffs is None:
        return None, None, None

    flat_coeffs = coeffs_to_array_2d(coeffs)
    sorted_mags = np.sort(np.abs(flat_coeffs))
    sorted_mags = sorted_mags[sorted_mags > 0]

    # Find threshold that gives target_pct retention
    target_percentile = 100.0 - target_pct
    threshold = (
        np.percentile(sorted_mags, target_percentile) if len(sorted_mags) > 0 else 0.0
    )

    thresholded = threshold_coefficients_2d(coeffs, float(threshold))
    X_recon = reconstruct_from_wavelets_2d(thresholded, wavelet)

    if X_recon is None or X_recon.shape != X_cpu.shape:
        return None, None, None

    X_recon_t = torch.from_numpy(X_recon).float().to(device)
    W_device = W.to(device)
    original_output_device = original_output.to(device)

    with torch.no_grad():
        output_2d = (W_device @ X_recon_t.T).T.float()

    snr_2d, noise_2d = _compute_snr(original_output_device, output_2d)

    output_residual = original_output_device - output_2d
    try:
        _, S_res, _ = torch.linalg.svd(output_residual.float(), full_matrices=False)
        resid_er_2d = _effective_rank(S_res)
    except:
        resid_er_2d = 0.0

    # Return: reconstructed activations and ACTIVATION residual for next stage
    # activation_residual = X - X_recon (in activation space, not output space)
    activation_residual = X.float().to(device) - X_recon_t
    return X_recon_t, activation_residual, (snr_2d, noise_2d, resid_er_2d)


def apply_2d_wavelet_energy_threshold(
    matrix, wavelet, energy_threshold=0.99, device="cuda"
):
    """Apply 2D wavelet decomposition and threshold to retain energy_threshold of energy.

    Returns:
        matrix_recon: Reconstructed matrix
        energy_pct_retained: Percentage of energy retained
    """
    matrix_cpu = matrix.float().cpu()

    coeffs, level = wavelet_decompose_2d(matrix_cpu, wavelet)
    if coeffs is None:
        return None, 0.0

    flat_coeffs = coeffs_to_array_2d(coeffs)
    mags = np.abs(flat_coeffs)

    # Sort by magnitude descending
    sorted_indices = np.argsort(-mags)
    sorted_mags = mags[sorted_indices]
    energy_cumsum = np.cumsum(sorted_mags**2)
    total_energy = energy_cumsum[-1]

    # Find threshold to retain energy_threshold
    target_energy = energy_threshold * total_energy
    num_needed = np.searchsorted(energy_cumsum, target_energy) + 1
    threshold = sorted_mags[num_needed - 1] if num_needed < len(sorted_mags) else 0.0

    # Threshold coefficients
    thresholded = threshold_coefficients_2d(coeffs, float(threshold))
    matrix_recon = reconstruct_from_wavelets_2d(thresholded, wavelet)

    if matrix_recon is None or matrix_recon.shape != matrix_cpu.shape:
        return None, 0.0

    energy_retained = energy_cumsum[num_needed - 1] / total_energy * 100.0
    return torch.from_numpy(matrix_recon).float().to(device), energy_retained


def apply_wavelets_mpo_hybrid(X, W, original_output, wavelet, device="cuda"):
    """Hybrid compression: wavelets on X and W, then MPO on W.

    Pipeline:
    1. X → X_wav (2D wavelets, 99% energy threshold)
    2. W → W_wav (2D wavelets, 99% energy threshold)
    3. W_wav → W_mpo (MPO decomposition for 50% param reduction)
    4. Reconstruct both wavelets
    5. Y = W_recon @ X_recon.T

    Returns:
        output: Final Y
        metrics: (snr, energy_pct, resid_er, cka, cos_sim, kl, ssim, sparsity_wav, x_energy_retained, w_energy_retained)
    """
    original_output_device = original_output.to(device)

    # Stage 1: 2D wavelets on X with 99% energy threshold
    X_recon, x_energy_retained = apply_2d_wavelet_energy_threshold(
        X, wavelet, energy_threshold=0.99, device=device
    )
    if X_recon is None:
        return None, None

    # Stage 2: 2D wavelets on W with 99% energy threshold
    W_recon, w_energy_retained = apply_2d_wavelet_energy_threshold(
        W, wavelet, energy_threshold=0.99, device=device
    )
    if W_recon is None:
        return None, None

    # Stage 3: MPO on W_recon (sparse weight matrix)
    cores, _ = _mpo_decompose_matrix(W_recon, rank_list=[1, 30, 1])
    if cores is None:
        return None, None

    W_mpo = _tt_reconstruct(cores)
    if W_mpo is None:
        return None, None

    W_mpo_device = W_mpo.float().to(device)

    # Compute final output: Y = W_mpo @ X_recon.T
    with torch.no_grad():
        final_output = (W_mpo_device @ X_recon.T).float()

    # Compute metrics
    snr, noise = _compute_snr(original_output_device, final_output)
    cka = _cka(original_output_device, final_output)
    cos_sim = _cosine_similarity_mean(original_output_device, final_output)
    kl = _kl_divergence(original_output_device, final_output)
    ssim = _ssim(original_output_device, final_output)

    residual = original_output_device - final_output
    _, S_res, _ = torch.linalg.svd(residual.float(), full_matrices=False)
    er = _effective_rank(S_res)
    energy = (S_res**2).sum().item()

    # Energy baseline for comparison
    _, S_baseline, _ = torch.linalg.svd(
        original_output_device.float(), full_matrices=False
    )
    energy_baseline = (S_baseline**2).sum().item()
    energy_pct = 100.0 * energy / energy_baseline

    sparsity_wav = _residual_sparsity_wavelet(residual, wavelet)

    return final_output, (
        snr,
        energy_pct,
        er,
        cka,
        cos_sim,
        kl,
        ssim,
        sparsity_wav,
        x_energy_retained,
        w_energy_retained,
    )


def apply_1d_wavelet_to_residual(
    residual_X, W, original_output, wavelet, target_pct=25.0, device="cuda"
):
    """Apply 1D wavelet to residual activations (in activation space).

    residual_X: X - X_2d (activation-space residual)
    Returns: W @ (1D reconstruction of residual)
    """
    residual_X_cpu = residual_X.float().cpu()
    in_features = residual_X_cpu.shape[1]
    num_samples = residual_X_cpu.shape[0]

    all_coeffs = []
    for feat_idx in range(in_features):
        signal = residual_X_cpu[:, feat_idx]
        coeffs, level = wavelet_decompose_1d(signal, wavelet)
        if coeffs is None:
            continue
        all_coeffs.append(coeffs)

    if not all_coeffs:
        return None, None

    # Find threshold for target_pct
    all_flat_coeffs = []
    for feat_idx in range(in_features):
        flat = coeffs_to_array_1d(all_coeffs[feat_idx])
        all_flat_coeffs.append(flat)
    all_flat_coeffs_concat = np.concatenate(all_flat_coeffs)

    sorted_mags = np.sort(np.abs(all_flat_coeffs_concat))
    sorted_mags = sorted_mags[sorted_mags > 0]
    target_percentile = 100.0 - target_pct
    threshold = (
        np.percentile(sorted_mags, target_percentile) if len(sorted_mags) > 0 else 0.0
    )

    X_recon_1d_list = []
    for feat_idx in range(in_features):
        coeffs = all_coeffs[feat_idx]
        thresholded = threshold_coefficients_1d(coeffs, float(threshold))
        signal_recon = reconstruct_from_wavelets_1d(thresholded, wavelet)

        if signal_recon is None:
            signal_recon = np.zeros(num_samples)
        else:
            if len(signal_recon) > num_samples:
                signal_recon = signal_recon[:num_samples]
            elif len(signal_recon) < num_samples:
                signal_recon = np.pad(
                    signal_recon, (0, num_samples - len(signal_recon))
                )

        X_recon_1d_list.append(signal_recon)

    X_recon_1d = np.column_stack(X_recon_1d_list)
    X_recon_1d_t = torch.from_numpy(X_recon_1d).float().to(device)

    W_device = W.to(device)
    original_output_device = original_output.to(device)

    with torch.no_grad():
        output_1d_residual = (W_device @ X_recon_1d_t.T).T.float()

    snr_1d_res, noise_1d_res = _compute_snr(original_output_device, output_1d_residual)

    residual_after_1d = original_output_device - output_1d_residual
    try:
        _, S_res, _ = torch.linalg.svd(residual_after_1d.float(), full_matrices=False)
        resid_er_1d = _effective_rank(S_res)
    except:
        resid_er_1d = 0.0

    return output_1d_residual, (snr_1d_res, noise_1d_res, resid_er_1d)


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Compute RCM column permutation
    print("Computing RCM column permutation...")
    X_cpu = X.float().cpu()
    cov = (X_cpu.T @ X_cpu) / X_cpu.shape[0]
    adj = cov.abs().numpy()
    threshold = np.percentile(adj, 95)
    adj[adj < threshold] = 0
    sparse_adj = csr_matrix(adj)
    col_perm = reverse_cuthill_mckee(sparse_adj).copy()

    X_rcm = X[:, col_perm]
    W_rcm = W[:, col_perm]
    with torch.no_grad():
        original_output_rcm = (W_rcm.float() @ X_rcm.float().T).T.float()
    print(
        f"RCM permutation computed. Verifying output matches: "
        f"{torch.allclose(original_output, original_output_rcm, atol=1e-4)}\n"
    )

    wavelets = ["haar", "db2", "db4"]

    print(f"{'='*100}")
    print(f"Cascaded Wavelet Compression: 2D (25%) + 1D (25%) on Residual")
    print(f"{'='*100}\n")

    original_output_device = original_output.to(device)

    for wavelet in wavelets:
        print(f"{wavelet}:")
        print(
            f"  {'Stage':20s} {'SNR':>10s} {'ER Real':>10s} {'Wav Sparsity':>12s} {'Energy %':>10s} {'CKA':>10s} {'Cos Sim':>10s} {'KL Div':>10s} {'SSIM':>10s}"
        )
        print(f"  {'-'*155}")

        # Baseline: no compression (original residual)
        _, S_baseline, _ = torch.linalg.svd(
            original_output_device.float(), full_matrices=False
        )
        er_baseline_real = _effective_rank(S_baseline)
        sparsity_baseline_wav = _residual_sparsity_wavelet(
            original_output_device, wavelet
        )
        energy_baseline = (S_baseline**2).sum().item()
        cka_baseline = _cka(original_output_device, original_output_device)
        cos_baseline = _cosine_similarity_mean(
            original_output_device, original_output_device
        )
        ssim_baseline = _ssim(original_output_device, original_output_device)
        print(
            f"  {'Baseline':20s} {'N/A':>10s} {er_baseline_real:10.0f} {sparsity_baseline_wav:11.1f}% {'100.0':>9s}% {cka_baseline:10.4f} {cos_baseline:10.4f} {'0.0000':>10s} {ssim_baseline:10.4f}"
        )

        # Stage 1: 2D wavelet at 25%
        X_recon_2d, residual_X, info_2d = apply_2d_wavelet(
            X, W, original_output, wavelet, 25.0, device
        )
        if X_recon_2d is not None:
            snr_2d, noise_2d, resid_er_2d = info_2d
            with torch.no_grad():
                output_2d = (W.to(device) @ X_recon_2d.T).T.float()

            # Compute residual ER and sparsity after 2D
            residual_2d = original_output_device - output_2d
            _, S_2d, _ = torch.linalg.svd(residual_2d.float(), full_matrices=False)
            er_after_2d_real = _effective_rank(S_2d)
            sparsity_2d_wav = _residual_sparsity_wavelet(residual_2d, wavelet)
            energy_after_2d = (S_2d**2).sum().item()
            energy_pct_2d = 100.0 * energy_after_2d / energy_baseline

            cka_2d = _cka(original_output_device, output_2d)
            cos_2d = _cosine_similarity_mean(original_output_device, output_2d)
            kl_2d = _kl_divergence(original_output_device, output_2d)
            ssim_2d = _ssim(original_output_device, output_2d)
            print(
                f"  {'2D wavelet (25%)':20s} {snr_2d:10.2f}dB {er_after_2d_real:10.0f} {sparsity_2d_wav:11.1f}% {energy_pct_2d:9.1f}% {cka_2d:10.4f} {cos_2d:10.4f} {kl_2d:10.4f} {ssim_2d:10.4f}"
            )
        else:
            print(f"  {'2D wavelet (25%)':20s} Failed")
            continue

        # Stage 2: 1D wavelet on residual at 25%
        output_1d_residual, info_1d = apply_1d_wavelet_to_residual(
            residual_X, W, original_output, wavelet, 25.0, device
        )
        if output_1d_residual is not None:
            snr_1d, noise_1d, resid_er_1d = info_1d
            # Compute combined output in OUTPUT space (both are outputs now)
            with torch.no_grad():
                output_2d = (W.to(device) @ X_recon_2d.T).T.float()
                total_output = output_2d + output_1d_residual
            snr_total, noise_total = _compute_snr(original_output_device, total_output)

            # Compute residual ER and sparsity after cascaded
            resid_total = original_output_device - total_output
            _, S_tot, _ = torch.linalg.svd(resid_total.float(), full_matrices=False)
            resid_er_total_real = _effective_rank(S_tot)
            sparsity_total_wav = _residual_sparsity_wavelet(resid_total, wavelet)
            energy_after_total = (S_tot**2).sum().item()
            energy_pct_total = 100.0 * energy_after_total / energy_baseline

            cka_total = _cka(original_output_device, total_output)
            cos_total = _cosine_similarity_mean(original_output_device, total_output)
            kl_total = _kl_divergence(original_output_device, total_output)
            ssim_total = _ssim(original_output_device, total_output)
            print(
                f"  {'1D on residual (25%)':20s} {snr_total:10.2f}dB {resid_er_total_real:10.0f} {sparsity_total_wav:11.1f}% {energy_pct_total:9.1f}% {cka_total:10.4f} {cos_total:10.4f} {kl_total:10.4f} {ssim_total:10.4f}"
            )
        else:
            print(f"  {'1D on residual (25%)':20s} Failed")

        print()

    # Comparison to single approaches
    print(f"\n{'='*165}")
    print("Comparison to Single Approaches at 50%")
    print(f"{'='*165}\n")
    print(
        f"{'Wavelet':10s} {'Method':20s} {'SNR':>10s} {'ER Real':>10s} {'Wav Sparsity':>12s} {'Energy %':>10s} {'CKA':>10s} {'Cos Sim':>10s} {'KL Div':>10s} {'SSIM':>10s}"
    )
    print(f"{'-'*155}")

    # Compute baseline energy once
    _, S_baseline_comp, _ = torch.linalg.svd(
        original_output_device.float(), full_matrices=False
    )
    energy_baseline_comp = (S_baseline_comp**2).sum().item()

    for wavelet in wavelets:
        # 2D alone at 50%
        X_recon_2d_50, _, info_2d_50 = apply_2d_wavelet(
            X, W, original_output, wavelet, 50.0, device
        )
        if info_2d_50:
            snr_2d_50, _, resid_er_2d_50 = info_2d_50
            with torch.no_grad():
                output_2d_50 = (W.to(device) @ X_recon_2d_50.T).T.float()
            resid_2d_50 = original_output_device - output_2d_50
            _, S_2d_50, _ = torch.linalg.svd(resid_2d_50.float(), full_matrices=False)
            er_2d_50_real = _effective_rank(S_2d_50)
            sparsity_2d_50_wav = _residual_sparsity_wavelet(resid_2d_50, wavelet)
            energy_2d_50 = (S_2d_50**2).sum().item()
            energy_pct_2d_50 = 100.0 * energy_2d_50 / energy_baseline_comp
            cka_2d_50 = _cka(original_output_device, output_2d_50)
            cos_2d_50 = _cosine_similarity_mean(original_output_device, output_2d_50)
            kl_2d_50 = _kl_divergence(original_output_device, output_2d_50)
            ssim_2d_50 = _ssim(original_output_device, output_2d_50)
            print(
                f"{wavelet:10s} {'2D alone (50%)':20s} {snr_2d_50:10.2f}dB {er_2d_50_real:10.0f} {sparsity_2d_50_wav:11.1f}% {energy_pct_2d_50:9.1f}% {cka_2d_50:10.4f} {cos_2d_50:10.4f} {kl_2d_50:10.4f} {ssim_2d_50:10.4f}"
            )

        # Cascaded at 25%+25%
        X_recon_2d, residual_X, info_2d = apply_2d_wavelet(
            X, W, original_output, wavelet, 25.0, device
        )
        if X_recon_2d is not None:
            output_1d_residual, info_1d = apply_1d_wavelet_to_residual(
                residual_X, W, original_output, wavelet, 25.0, device
            )
            if output_1d_residual is not None:
                with torch.no_grad():
                    output_2d = (W.to(device) @ X_recon_2d.T).T.float()
                    total_output = output_2d + output_1d_residual
                snr_cascade, _ = _compute_snr(original_output_device, total_output)
                resid_cascade = original_output_device - total_output
                _, S_casc, _ = torch.linalg.svd(
                    resid_cascade.float(), full_matrices=False
                )
                er_cascade_real = _effective_rank(S_casc)
                sparsity_cascade_wav = _residual_sparsity_wavelet(
                    resid_cascade, wavelet
                )
                energy_cascade = (S_casc**2).sum().item()
                energy_pct_cascade = 100.0 * energy_cascade / energy_baseline_comp
                cka_cascade = _cka(original_output_device, total_output)
                cos_cascade = _cosine_similarity_mean(
                    original_output_device, total_output
                )
                kl_cascade = _kl_divergence(original_output_device, total_output)
                ssim_cascade = _ssim(original_output_device, total_output)
                print(
                    f"{wavelet:10s} {'Cascaded (25%+25%)':20s} {snr_cascade:10.2f}dB {er_cascade_real:10.0f} {sparsity_cascade_wav:11.1f}% {energy_pct_cascade:9.1f}% {cka_cascade:10.4f} {cos_cascade:10.4f} {kl_cascade:10.4f} {ssim_cascade:10.4f}"
                )

        print()

    # RCM-reordered cascade
    print(f"\n{'='*100}")
    print(f"With RCM Column Reordering: 2D (25%) + 1D (25%) on Residual")
    print(f"{'='*100}\n")

    for wavelet in wavelets:
        print(f"{wavelet}:")
        print(
            f"  {'Stage':20s} {'SNR':>10s} {'ER Real':>10s} {'Wav Sparsity':>12s} {'Energy %':>10s} {'CKA':>10s} {'Cos Sim':>10s} {'KL Div':>10s} {'SSIM':>10s}"
        )
        print(f"  {'-'*155}")

        _, S_baseline_rcm, _ = torch.linalg.svd(
            original_output_device.float(), full_matrices=False
        )
        er_baseline_rcm = _effective_rank(S_baseline_rcm)
        sparsity_baseline_rcm = _residual_sparsity_wavelet(
            original_output_device, wavelet
        )
        energy_baseline_rcm = (S_baseline_rcm**2).sum().item()
        print(
            f"  {'Baseline':20s} {'N/A':>10s} {er_baseline_rcm:10.0f} {sparsity_baseline_rcm:11.1f}% {'100.0':>9s}% {'1.0000':>10s} {'1.0000':>10s} {'0.0000':>10s} {'1.0000':>10s}"
        )

        X_recon_2d_rcm, residual_X_rcm, info_2d_rcm = apply_2d_wavelet(
            X_rcm, W_rcm, original_output, wavelet, 25.0, device
        )
        if X_recon_2d_rcm is not None:
            snr_2d_rcm, _, _ = info_2d_rcm
            with torch.no_grad():
                output_2d_rcm = (W_rcm.to(device) @ X_recon_2d_rcm.T).T.float()

            residual_2d_rcm = original_output_device - output_2d_rcm
            _, S_2d_rcm, _ = torch.linalg.svd(
                residual_2d_rcm.float(), full_matrices=False
            )
            er_2d_rcm = _effective_rank(S_2d_rcm)
            sparsity_2d_rcm = _residual_sparsity_wavelet(residual_2d_rcm, wavelet)
            energy_2d_rcm = (S_2d_rcm**2).sum().item()
            energy_pct_2d_rcm = 100.0 * energy_2d_rcm / energy_baseline_rcm

            cka_2d_rcm = _cka(original_output_device, output_2d_rcm)
            cos_2d_rcm = _cosine_similarity_mean(original_output_device, output_2d_rcm)
            kl_2d_rcm = _kl_divergence(original_output_device, output_2d_rcm)
            ssim_2d_rcm = _ssim(original_output_device, output_2d_rcm)
            print(
                f"  {'2D wavelet (25%)':20s} {snr_2d_rcm:10.2f}dB {er_2d_rcm:10.0f} {sparsity_2d_rcm:11.1f}% {energy_pct_2d_rcm:9.1f}% {cka_2d_rcm:10.4f} {cos_2d_rcm:10.4f} {kl_2d_rcm:10.4f} {ssim_2d_rcm:10.4f}"
            )
        else:
            print(f"  {'2D wavelet (25%)':20s} Failed")
            continue

        output_1d_rcm, info_1d_rcm = apply_1d_wavelet_to_residual(
            residual_X_rcm, W_rcm, original_output, wavelet, 25.0, device
        )
        if output_1d_rcm is not None:
            snr_1d_rcm, _, _ = info_1d_rcm
            with torch.no_grad():
                output_2d_rcm = (W_rcm.to(device) @ X_recon_2d_rcm.T).T.float()
                total_output_rcm = output_2d_rcm + output_1d_rcm
            snr_total_rcm, _ = _compute_snr(original_output_device, total_output_rcm)

            resid_total_rcm = original_output_device - total_output_rcm
            _, S_tot_rcm, _ = torch.linalg.svd(
                resid_total_rcm.float(), full_matrices=False
            )
            er_total_rcm = _effective_rank(S_tot_rcm)
            sparsity_total_rcm = _residual_sparsity_wavelet(resid_total_rcm, wavelet)
            energy_total_rcm = (S_tot_rcm**2).sum().item()
            energy_pct_total_rcm = 100.0 * energy_total_rcm / energy_baseline_rcm

            cka_total_rcm = _cka(original_output_device, total_output_rcm)
            cos_total_rcm = _cosine_similarity_mean(
                original_output_device, total_output_rcm
            )
            kl_total_rcm = _kl_divergence(original_output_device, total_output_rcm)
            ssim_total_rcm = _ssim(original_output_device, total_output_rcm)
            print(
                f"  {'1D on residual (25%)':20s} {snr_total_rcm:10.2f}dB {er_total_rcm:10.0f} {sparsity_total_rcm:11.1f}% {energy_pct_total_rcm:9.1f}% {cka_total_rcm:10.4f} {cos_total_rcm:10.4f} {kl_total_rcm:10.4f} {ssim_total_rcm:10.4f}"
            )
        else:
            print(f"  {'1D on residual (25%)':20s} Failed")

        print()

    # RCM comparison at 50%
    print(f"\n{'='*165}")
    print("Comparison: Original vs RCM at 50%")
    print(f"{'='*165}\n")
    print(
        f"{'Wavelet':10s} {'Method':20s} {'SNR':>10s} {'ER Real':>10s} {'Wav Sparsity':>12s} {'Energy %':>10s} {'CKA':>10s} {'Cos Sim':>10s} {'KL Div':>10s} {'SSIM':>10s}"
    )
    print(f"{'-'*155}")

    _, S_bl, _ = torch.linalg.svd(original_output_device.float(), full_matrices=False)
    energy_bl = (S_bl**2).sum().item()

    for wavelet in wavelets:
        # Original 2D at 50%
        X_recon_orig, _, info_orig = apply_2d_wavelet(
            X, W, original_output, wavelet, 50.0, device
        )
        if info_orig:
            snr_orig, _, _ = info_orig
            with torch.no_grad():
                out_orig = (W.to(device) @ X_recon_orig.T).T.float()
            res_orig = original_output_device - out_orig
            _, S_orig, _ = torch.linalg.svd(res_orig.float(), full_matrices=False)
            er_orig = _effective_rank(S_orig)
            sp_orig = _residual_sparsity_wavelet(res_orig, wavelet)
            en_orig = 100.0 * (S_orig**2).sum().item() / energy_bl
            cka_orig = _cka(original_output_device, out_orig)
            cos_orig = _cosine_similarity_mean(original_output_device, out_orig)
            kl_orig = _kl_divergence(original_output_device, out_orig)
            ssim_orig = _ssim(original_output_device, out_orig)
            print(
                f"{wavelet:10s} {'Original 2D (50%)':20s} {snr_orig:10.2f}dB {er_orig:10.0f} {sp_orig:11.1f}% {en_orig:9.1f}% {cka_orig:10.4f} {cos_orig:10.4f} {kl_orig:10.4f} {ssim_orig:10.4f}"
            )

        # RCM 2D at 50%
        X_recon_rcm50, _, info_rcm50 = apply_2d_wavelet(
            X_rcm, W_rcm, original_output, wavelet, 50.0, device
        )
        if info_rcm50:
            snr_rcm50, _, _ = info_rcm50
            with torch.no_grad():
                out_rcm50 = (W_rcm.to(device) @ X_recon_rcm50.T).T.float()
            res_rcm50 = original_output_device - out_rcm50
            _, S_rcm50, _ = torch.linalg.svd(res_rcm50.float(), full_matrices=False)
            er_rcm50 = _effective_rank(S_rcm50)
            sp_rcm50 = _residual_sparsity_wavelet(res_rcm50, wavelet)
            en_rcm50 = 100.0 * (S_rcm50**2).sum().item() / energy_bl
            cka_rcm50 = _cka(original_output_device, out_rcm50)
            cos_rcm50 = _cosine_similarity_mean(original_output_device, out_rcm50)
            kl_rcm50 = _kl_divergence(original_output_device, out_rcm50)
            ssim_rcm50 = _ssim(original_output_device, out_rcm50)
            print(
                f"{wavelet:10s} {'RCM 2D (50%)':20s} {snr_rcm50:10.2f}dB {er_rcm50:10.0f} {sp_rcm50:11.1f}% {en_rcm50:9.1f}% {cka_rcm50:10.4f} {cos_rcm50:10.4f} {kl_rcm50:10.4f} {ssim_rcm50:10.4f}"
            )

        print()

    # Wavelets (99% energy) + MPO hybrid approach
    print(f"\n{'='*180}")
    print("Wavelets (99% energy on X & W) + MPO on W Hybrid")
    print(f"{'='*180}\n")
    print(
        f"{'Wavelet':10s} {'SNR':>10s} {'ER Real':>10s} {'Wav Sparsity':>12s} {'Energy %':>10s} {'X Energy':>10s} {'W Energy':>10s} {'CKA':>10s} {'KL Div':>10s}"
    )
    print(f"{'-'*170}")

    for wavelet in wavelets:
        final_output, metrics = apply_wavelets_mpo_hybrid(
            X, W, original_output, wavelet, device
        )
        if metrics:
            (
                snr,
                energy_pct,
                er,
                cka,
                cos_sim,
                kl,
                ssim,
                sparsity_wav,
                x_energy,
                w_energy,
            ) = metrics
            print(
                f"{wavelet:10s} {snr:10.2f}dB {er:10.0f} {sparsity_wav:11.1f}% {energy_pct:9.1f}% {x_energy:10.1f}% {w_energy:10.1f}% {cka:10.4f} {kl:10.4f}"
            )
        else:
            print(f"{wavelet:10s} Failed")


if __name__ == "__main__":
    main()
