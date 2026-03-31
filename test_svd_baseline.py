"""Quick test: What SNR does pure SVD achieve at various ranks?"""
import torch
from transformers import AutoModelForCausalLM

def compute_sqnr(original, approximation):
    """Compute SQNR in dB."""
    signal_power = torch.var(original)
    mse_noise = torch.mean((original - approximation) ** 2)
    sqnr_linear = signal_power / (mse_noise + 1e-10)
    sqnr_db = 10 * torch.log10(sqnr_linear)
    return sqnr_db.item()

print("="*80)
print("SVD Baseline: What SNR does pure low-rank approximation achieve?")
print("="*80)

# Load real model weights
print("\nLoading meta-llama/Llama-3.2-1B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

q_proj = model.model.layers[0].self_attn.q_proj
W = q_proj.weight.data.float().clone()
print(f"Loaded q_proj weight: {W.shape}")
print(f"Original params: {W.numel():,}")

del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# SVD
print("\nPerforming SVD...")
U, S, Vh = torch.linalg.svd(W, full_matrices=False)

print(f"\n{'Rank':<8} {'SNR':<12} {'Params':<15} {'Ratio':<10} {'Target?'}")
print("-"*70)

# Test various ranks
ranks = [100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
for rank in ranks:
    # Low-rank approximation
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    W_approx = U_r @ torch.diag(S_r) @ Vh_r

    # Compute SNR
    snr = compute_sqnr(W, W_approx)

    # Params: rank * (m + n)
    params = rank * (W.shape[0] + W.shape[1])
    param_ratio = params / W.numel()

    target_mark = "✓" if snr >= 35.0 and param_ratio < 1.0 else ""

    print(f"{rank:<8} {snr:>6.2f} dB   {params:>10,}    {param_ratio:>6.2f}x   {target_mark}")

print("\n" + "="*80)
print("Conclusion:")
print("  - Pure SVD provides a baseline for achievable compression")
print("  - If SVD can't reach 35 dB with < 1.0x params, hybrid likely can't either")
print("  - BlockTensorized + residual can at best match (not exceed) SVD quality")
print("="*80)
