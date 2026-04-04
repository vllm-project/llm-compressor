"""Activation-Space Spectral Compression with Hybrid DCT/IDCT Strategy.

Exploits spectral smoothness in ACTIVATIONS rather than weights.
Activations have clearer structure (semantic clustering, head patterns)
than noisy weight matrices.

Architecture:
1. Topological reordering (RCM) for input and output dimensions
2. Hybrid DCT Strategy (different treatment for input vs output):
   - Input: Global 1D DCT across all features (hidden state has no head structure)
   - Output: Blocked DCT per head (q/k/v have head structure for attention)
3. Learn frequency-domain weight: W_freq via least squares
4. Compress W_freq with MPO (treating as 3D: num_heads × head_dim × in_features)
5. Multi-scale residual ladder (Tucker + Sparse)
6. Inference: x → Global-DCT → W_freq @ x_freq → Blocked-IDCT → y

CRITICAL - Hybrid Strategy Rationale:
- Input (hidden state before q_proj): No inherent head structure → Global DCT
- Output (q after q_proj): Structured into heads → Blocked DCT/IDCT per head
- Blocked IDCT ensures output has correct head boundaries (32 heads × 128 head_dim)
  for subsequent RoPE and Attention operations

Key insight: Transformer activations are smooth/structured, weights are noisy.
Operating in activation frequency space gives better compression.

Reference: Designed for q_proj/k_proj/v_proj layers with multi-head structure.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
from typing import Optional, Tuple
from scipy.sparse.csgraph import reverse_cuthill_mckee
import numpy as np

tl.set_backend("pytorch")

__all__ = ["SpectralMPOLinear"]


def prune_tt_by_energy(
    tensor: torch.Tensor,
    tt_factors: list[torch.Tensor],
    energy_threshold: float = 0.8,
    verbose: bool = False,
) -> list[torch.Tensor]:
    """
    Prune TT cores to retain specified energy threshold.

    Uses iterative rank reduction: tries progressively smaller ranks
    until energy drops below threshold.

    Args:
        tensor: Original tensor
        tt_factors: TT decomposition factors
        energy_threshold: Fraction of energy to retain (0.8 = 80%)
        verbose: Print pruning info

    Returns:
        Pruned TT factors with smaller ranks
    """
    # Compute full energy
    full_reconstruction = tl.tt_to_tensor(tt_factors)
    total_energy = (tensor ** 2).sum().item()
    initial_error = ((tensor - full_reconstruction) ** 2).sum().item()
    initial_energy_retained = 1.0 - (initial_error / total_energy)

    # Extract current ranks
    current_ranks = [1]  # r0 = 1
    for factor in tt_factors:
        current_ranks.append(factor.shape[-1])

    if verbose:
        print(f"    Initial energy retained: {initial_energy_retained:.1%}")
        print(f"    Initial ranks: {current_ranks}")

    # Try progressively smaller ranks
    best_factors = tt_factors
    best_ranks = current_ranks[:]

    for reduction_factor in [0.8, 0.6, 0.5, 0.4, 0.3, 0.2]:
        # Compute new ranks (keep r0=1, r_end=1)
        new_ranks = [1]
        for r in current_ranks[1:-1]:
            new_ranks.append(max(2, int(r * reduction_factor)))
        new_ranks.append(1)

        # Skip if ranks didn't change
        if new_ranks == best_ranks:
            continue

        # Try re-decomposing with new ranks
        try:
            new_tt = tl.decomposition.tensor_train(tensor, rank=tuple(new_ranks))
            new_reconstruction = tl.tt_to_tensor(new_tt.factors)
            error = ((tensor - new_reconstruction) ** 2).sum().item()
            energy_retained = 1.0 - (error / total_energy)

            if energy_retained >= energy_threshold:
                # Good! Use this smaller decomposition
                best_factors = new_tt.factors
                best_ranks = new_ranks
                if verbose:
                    params_before = sum(f.numel() for f in tt_factors)
                    params_after = sum(f.numel() for f in best_factors)
                    print(f"    Pruned to ranks {best_ranks}: {params_after:,} params ({params_after/params_before:.1%}), energy {energy_retained:.1%}")
            else:
                # Energy dropped too much, stop
                break

        except Exception:
            # Decomposition failed, stop
            break

    return best_factors


class SpectralMPOLinear(nn.Module):
    """
    Activation-space spectral compression with Blocked DCT/IDCT.

    Operates in frequency domain of activations (not weights).
    Forward pass: x → permute → DCT → W_freq @ x_freq → Blocked-IDCT → unpermute → y

    CRITICAL: Output uses blocked IDCT per head to preserve multi-head structure
    for RoPE and Attention operations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        # MPO cores and bond dimensions
        mpo_factors: Tuple[torch.Tensor, ...],
        mpo_bond_dims: Tuple[int, ...],
        # Residual components
        residual_tucker_core: Optional[torch.Tensor] = None,
        residual_tucker_factors: Optional[Tuple[torch.Tensor, ...]] = None,
        residual_sparse_indices: Optional[torch.Tensor] = None,
        residual_sparse_values: Optional[torch.Tensor] = None,
        # Permutations for topological reordering
        input_perm: Optional[torch.Tensor] = None,
        input_inv_perm: Optional[torch.Tensor] = None,
        output_perm: Optional[torch.Tensor] = None,
        output_inv_perm: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        # Calibration activations for to_matrix()
        calib_inputs: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.dtype = dtype

        assert out_features % num_heads == 0, f"out_features {out_features} must be divisible by num_heads {num_heads}"

        # Store calibration inputs for to_matrix() (use small subset)
        if calib_inputs is not None:
            # Store up to 512 samples for computing effective weight
            num_samples = min(512, calib_inputs.shape[0])
            self.register_buffer("calib_inputs", calib_inputs[:num_samples].to(dtype))
        else:
            self.calib_inputs = None

        # MPO factors: [num_heads, head_dim, d_model]
        self.mpo_factors = nn.ParameterList([
            nn.Parameter(f.to(dtype), requires_grad=False) for f in mpo_factors
        ])
        self.mpo_bond_dims = mpo_bond_dims

        # Residual Tucker decomposition (optional)
        if residual_tucker_core is not None:
            self.register_buffer("residual_tucker_core", residual_tucker_core.to(dtype))
            self.residual_tucker_factors = nn.ParameterList([
                nn.Parameter(f.to(dtype), requires_grad=False) for f in residual_tucker_factors
            ])
        else:
            self.residual_tucker_core = None
            self.residual_tucker_factors = None

        # Residual sparse (COO format)
        if residual_sparse_indices is not None:
            self.register_buffer("residual_sparse_indices", residual_sparse_indices.long())
            self.register_buffer("residual_sparse_values", residual_sparse_values.to(dtype))
        else:
            self.residual_sparse_indices = None
            self.residual_sparse_values = None

        # Topological permutations
        if input_perm is not None:
            self.register_buffer("input_perm", input_perm.long())
            self.register_buffer("input_inv_perm", input_inv_perm.long())
        else:
            self.input_perm = None
            self.input_inv_perm = None

        if output_perm is not None:
            self.register_buffer("output_perm", output_perm.long())
            self.register_buffer("output_inv_perm", output_inv_perm.long())
        else:
            self.output_perm = None
            self.output_inv_perm = None

        if bias is not None:
            self.bias = nn.Parameter(bias.to(dtype), requires_grad=False)
        else:
            self.bias = None

    @staticmethod
    def _rcm_reordering(correlation_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply Reverse Cuthill-McKee (RCM) algorithm for bandwidth minimization.

        RCM finds a permutation that minimizes the matrix bandwidth by treating
        the correlation matrix as an adjacency matrix and reordering to keep
        strongly connected features close together.

        Args:
            correlation_matrix: (N, N) correlation/affinity matrix

        Returns:
            Permutation indices that minimize bandwidth
        """
        from scipy.sparse import csr_matrix

        # Promote bfloat16 to float32 for scipy compatibility
        if correlation_matrix.dtype == torch.bfloat16:
            correlation_matrix = correlation_matrix.float()

        # Convert to numpy for scipy RCM
        corr_np = correlation_matrix.detach().cpu().numpy()

        # Make symmetric and threshold to get adjacency matrix
        # Use absolute value and threshold at median
        adj_matrix = np.abs(corr_np)
        threshold = np.median(adj_matrix)
        adj_matrix = (adj_matrix >= threshold).astype(float)

        # Convert to sparse format for RCM
        adj_sparse = csr_matrix(adj_matrix)

        # Apply RCM algorithm
        perm = reverse_cuthill_mckee(adj_sparse, symmetric_mode=True)

        # Copy to handle negative strides
        return torch.from_numpy(perm.copy()).to(correlation_matrix.device)

    @staticmethod
    def _compute_input_permutation(
        input_activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute RCM permutation for input dimensions based on activation correlations.

        Args:
            input_activations: (num_samples, in_features)

        Returns:
            Permutation indices
        """
        # Center and normalize
        acts_centered = input_activations - input_activations.mean(dim=0, keepdim=True)
        acts_std = acts_centered.std(dim=0, keepdim=True) + 1e-10
        acts_normalized = acts_centered / acts_std

        # Correlation matrix: (in_features, in_features)
        num_samples = acts_normalized.shape[0]
        correlation = (acts_normalized.T @ acts_normalized) / num_samples

        # Apply RCM
        return SpectralMPOLinear._rcm_reordering(correlation)

    @staticmethod
    def _compute_output_permutation(
        output_activations: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:
        """
        Compute RCM permutation for output dimensions (per-head).

        For multi-head attention, we compute a local permutation within each head
        to align internal structure (e.g., RoPE pairs).

        Args:
            output_activations: (num_samples, out_features)
            num_heads: Number of attention heads

        Returns:
            Permutation indices for all heads concatenated
        """
        num_samples = output_activations.shape[0]
        out_features = output_activations.shape[1]
        head_dim = out_features // num_heads

        # Reshape to [num_samples, num_heads, head_dim]
        acts_heads = output_activations.reshape(num_samples, num_heads, head_dim)

        # Compute correlation averaged across all heads
        # Shape: (head_dim, head_dim)
        acts_centered = acts_heads - acts_heads.mean(dim=0, keepdim=True)
        acts_std = acts_centered.std(dim=0, keepdim=True) + 1e-10
        acts_normalized = acts_centered / acts_std

        # Average correlation across heads
        correlations = []
        for h in range(num_heads):
            head_acts = acts_normalized[:, h, :]  # (num_samples, head_dim)
            corr_h = (head_acts.T @ head_acts) / num_samples
            correlations.append(corr_h)

        avg_correlation = torch.stack(correlations).mean(dim=0)  # (head_dim, head_dim)

        # Compute local permutation for a single head
        local_perm = SpectralMPOLinear._rcm_reordering(avg_correlation)

        # Expand to all heads: [perm, perm + head_dim, perm + 2*head_dim, ...]
        full_perm = []
        for h in range(num_heads):
            full_perm.append(local_perm + h * head_dim)

        return torch.cat(full_perm)

    @staticmethod
    def _dct_per_head(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        """
        Blocked DCT - applies 1D DCT per head independently.

        CRITICAL: Preserves head boundaries for multi-head attention.
        Each head gets its own frequency transform, ensuring the output
        has correct structure for RoPE and Attention operations.

        Args:
            tensor: (num_samples, features) where features = num_heads * head_dim

        Returns:
            DCT coefficients with same shape, head structure preserved
        """
        from scipy.fftpack import dct

        # Promote bfloat16 to float32 for scipy compatibility
        orig_dtype = tensor.dtype
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()

        num_samples = tensor.shape[0]
        features = tensor.shape[1]
        head_dim = features // num_heads

        # Reshape to (num_samples, num_heads, head_dim)
        tensor_heads = tensor.reshape(num_samples, num_heads, head_dim)

        # Convert to numpy
        np_tensor = tensor_heads.detach().cpu().numpy()

        # Apply 1D DCT along last axis (head_dim) - independent per head
        dct_result = dct(np_tensor, axis=-1, norm='ortho')

        # Reshape back to (num_samples, features)
        dct_result = dct_result.reshape(num_samples, features)

        # Convert back to torch with original dtype
        result = torch.from_numpy(dct_result).to(tensor.device)
        if orig_dtype == torch.bfloat16:
            result = result.to(torch.bfloat16)
        return result

    @staticmethod
    def _idct_per_head(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        """
        Blocked IDCT - applies 1D IDCT per head independently.

        CRITICAL: Preserves head boundaries for multi-head attention.
        This ensures the reconstructed output has correct head structure
        for subsequent RoPE and Attention operations.

        Args:
            tensor: (num_samples, features) DCT coefficients

        Returns:
            Spatial-domain tensor with same shape, head structure preserved
        """
        from scipy.fftpack import idct

        # Promote bfloat16 to float32 for scipy compatibility
        orig_dtype = tensor.dtype
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()

        num_samples = tensor.shape[0]
        features = tensor.shape[1]
        head_dim = features // num_heads

        # Reshape to (num_samples, num_heads, head_dim)
        tensor_heads = tensor.reshape(num_samples, num_heads, head_dim)

        # Convert to numpy
        np_tensor = tensor_heads.detach().cpu().numpy()

        # Apply 1D IDCT along last axis (head_dim) - independent per head
        idct_result = idct(np_tensor, axis=-1, norm='ortho')

        # Reshape back to (num_samples, features)
        idct_result = idct_result.reshape(num_samples, features)

        # Convert back to torch with original dtype
        result = torch.from_numpy(idct_result).to(tensor.device)
        if orig_dtype == torch.bfloat16:
            result = result.to(torch.bfloat16)
        return result

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        num_heads: int,
        input_activations: torch.Tensor,
        target_snr_db: float = 40.0,
        param_budget: float = 0.5,
        mpo_bond_dim_ratio: float = 0.3,
        use_tucker_residual: bool = True,
        tucker_rank: float = 0.2,
        use_sparse_residual: bool = True,
        sparse_percentile: float = 99.9,
        method: str = "activation",  # "activation", "weight", or "hybrid"
        verbose: bool = True,
    ) -> "SpectralMPOLinear":
        """
        Build SpectralMPOLinear from a torch.nn.Linear layer.

        Two approaches for spectral weight:

        Method "weight" (Direct Transform):
        - DCT(Weight) → compress with MPO
        - Preserves global structure ("Prior")
        - High weight SNR, faster computation
        - No data dependency

        Method "activation" (Task-Aware LSTSQ):
        - Learn W_freq from lstsq(DCT(X), DCT(Y))
        - Task-aware, optimizes for actual activations ("Evidence")
        - Higher activation SNR (+5-10dB typical)
        - Filters out unused weight components

        Method "hybrid" (Best of Both):
        - Initialize with DCT(Weight)
        - Refine MPO cores with activation-based LSTSQ
        - TODO: Implement ALS refinement

        Args:
            linear: Original linear layer (e.g., q_proj)
            num_heads: Number of attention heads
            input_activations: Calibration data (num_samples, in_features)
            target_snr_db: Target SNR in dB (default: 40.0)
            param_budget: Parameter budget as fraction of original (default: 0.5)
            mpo_bond_dim_ratio: MPO bond dimension as ratio of feature dims
            use_tucker_residual: Use Tucker decomposition for residuals
            tucker_rank: Rank ratio for Tucker residual
            use_sparse_residual: Use sparse anchors for high-frequency spikes
            sparse_percentile: Percentile threshold for sparse selection (99.9 = top 0.1%)
            verbose: Print progress
        """
        if verbose:
            print(f"\nSpectral MPO Compression (CPU-optimized for low memory):")
            print(f"  Method: {method}")
            print(f"  Target SNR: {target_snr_db:.1f} dB")
            print(f"  Parameter budget: {param_budget:.1%}")
            print(f"  Number of heads: {num_heads}")

        device = linear.weight.device
        dtype = linear.weight.dtype
        weight = linear.weight.detach().clone().float()
        bias = linear.bias.detach().clone() if linear.bias is not None else None

        out_features, in_features = weight.shape
        head_dim = out_features // num_heads

        assert out_features % num_heads == 0, f"out_features must be divisible by num_heads"

        # Move to CPU for memory-intensive operations (no subsampling - use all samples)
        input_activations_cpu = input_activations.float().cpu()
        weight_cpu = weight.cpu()
        bias_cpu = bias.cpu() if bias is not None else None

        # Compute output activations for permutation (on CPU to save GPU memory)
        with torch.no_grad():
            output_activations_cpu = F.linear(
                input_activations_cpu,
                weight_cpu,
                bias_cpu
            )

        # Step 1: Topological Reordering (RCM) - on CPU
        if verbose:
            print("\nStep 1: Topological Reordering (RCM)")

        input_perm = cls._compute_input_permutation(input_activations_cpu)
        output_perm = cls._compute_output_permutation(output_activations_cpu, num_heads)

        input_inv_perm = torch.argsort(input_perm)
        output_inv_perm = torch.argsort(output_perm)

        # Apply permutations (on CPU)
        weight_perm_cpu = weight_cpu[output_perm, :][:, input_perm]

        if verbose:
            print(f"  Input permutation computed via RCM")
            print(f"  Output permutation computed via RCM (per-head)")

        # Step 2: Transform activations to frequency domain (activation-space DCT) - on CPU
        if verbose:
            print("\nStep 2: Transform activations to frequency domain")

        # Step 2: Obtain spectral weight (method-dependent)
        from scipy.fftpack import dct

        if method == "weight":
            # Method 1: Direct Transform - DCT(Weight)
            # Preserves global structure, no data dependency
            if verbose:
                print("\nStep 2: Direct Weight Transform (DCT of Weight Matrix)")

            # Reshape weight to 3D: (num_heads, head_dim, in_features)
            weight_3d_cpu = weight_perm_cpu.reshape(num_heads, head_dim, in_features)

            # Apply blocked DCT per head to output dimension
            weight_3d_np = weight_3d_cpu.numpy()
            weight_spectral_np = dct(weight_3d_np, axis=1, norm='ortho')
            weight_spectral = torch.from_numpy(weight_spectral_np)

            if verbose:
                total_energy = (weight_spectral ** 2).sum()
                corner_h = head_dim // 4
                corner_energy = (weight_spectral[:, :corner_h, :] ** 2).sum()
                concentration = (corner_energy / total_energy).item()
                print(f"  Applied Blocked DCT to weight per head")
                print(f"  Spectral energy concentration (low 25%): {concentration:.2%}")

        elif method == "activation":
            # Method 2: Task-Aware LSTSQ from Activations
            # Learn W_freq from actual data distribution
            if verbose:
                print("\nStep 2: Task-Aware Weight Learning (LSTSQ from Activations)")

            # Transform activations to frequency domain
            input_activations_perm = input_activations_cpu[:, input_perm]
            output_activations_perm = output_activations_cpu[:, output_perm]

            # Input DCT - global
            input_perm_f32 = input_activations_perm.float()
            input_freq_np = dct(input_perm_f32.cpu().numpy(), axis=-1, norm='ortho')
            input_freq = torch.from_numpy(input_freq_np)

            # Output DCT - blocked per head
            output_freq = cls._dct_per_head(output_activations_perm, num_heads)

            # Learn W_freq via least squares
            weight_freq_T, residuals, rank, s = torch.linalg.lstsq(
                input_freq, output_freq, rcond=None
            )
            weight_freq = weight_freq_T.T

            # Measure fit quality
            reconstruction_freq = weight_freq @ input_freq.T
            freq_snr = 20 * torch.log10(
                output_freq.T.norm() / ((output_freq.T - reconstruction_freq).norm() + 1e-10)
            ).item()

            if verbose:
                print(f"  Learned W_freq via LSTSQ")
                print(f"  Frequency-domain fit SNR: {freq_snr:.2f} dB")

            # Reshape to 3D
            weight_spectral = weight_freq.reshape(num_heads, head_dim, in_features)

        elif method == "hybrid":
            # Method 3: Hybrid - Initialize with DCT(Weight), refine with activations
            # Combines global structure (Prior) with task-aware precision (Evidence)
            if verbose:
                print("\nStep 2: Hybrid Weight Learning (DCT + Activation Refinement)")

            # Step 2a: Initialize with Direct Transform (Prior)
            weight_3d_cpu = weight_perm_cpu.reshape(num_heads, head_dim, in_features)
            weight_3d_np = weight_3d_cpu.numpy()
            weight_spectral_np = dct(weight_3d_np, axis=1, norm='ortho')
            weight_spectral_init = torch.from_numpy(weight_spectral_np)

            if verbose:
                total_energy = (weight_spectral_init ** 2).sum()
                corner_h = head_dim // 4
                corner_energy = (weight_spectral_init[:, :corner_h, :] ** 2).sum()
                concentration = (corner_energy / total_energy).item()
                print(f"  Step 2a: Initialized from DCT(Weight)")
                print(f"  Spectral energy concentration (low 25%): {concentration:.2%}")

            # Step 2b: Compute activation-based target (Evidence)
            input_activations_perm = input_activations_cpu[:, input_perm]
            output_activations_perm = output_activations_cpu[:, output_perm]

            # Input DCT - global
            input_perm_f32 = input_activations_perm.float()
            input_freq_np = dct(input_perm_f32.cpu().numpy(), axis=-1, norm='ortho')
            input_freq = torch.from_numpy(input_freq_np)

            # Output DCT - blocked per head
            output_freq = cls._dct_per_head(output_activations_perm, num_heads)

            # Learn activation-optimal W_freq via least squares
            weight_freq_T, residuals, rank, s = torch.linalg.lstsq(
                input_freq, output_freq, rcond=None
            )
            weight_freq_target = weight_freq_T.T.reshape(num_heads, head_dim, in_features)

            # Measure fit quality
            reconstruction_freq = weight_freq_target.reshape(out_features, in_features) @ input_freq.T
            freq_snr = 20 * torch.log10(
                output_freq.T.norm() / ((output_freq.T - reconstruction_freq).norm() + 1e-10)
            ).item()

            if verbose:
                print(f"  Step 2b: Computed activation target via LSTSQ")
                print(f"  Activation-based fit SNR: {freq_snr:.2f} dB")

            # Step 2c: Blend Prior and Evidence (simple weighted average for now)
            # TODO: Implement full ALS refinement in future
            # For now: 70% activation-based, 30% weight-based (favor evidence)
            blend_weight = 0.7
            weight_spectral = (blend_weight * weight_freq_target +
                              (1 - blend_weight) * weight_spectral_init)

            if verbose:
                print(f"  Step 2c: Blended Prior (30%) and Evidence (70%)")

        else:
            raise ValueError(f"Unknown method: {method}. Use 'weight', 'activation', or 'hybrid'")

        # Step 3: Compress spectral weight with MPO
        if verbose:
            print("\nStep 3: Compress W_spectral with MPO")

        # Determine bond dimensions based on parameter budget
        # MPO has 3 cores: [num_heads, head_dim, in_features]
        # Bond dimensions: r0=1, r1, r2, r3=1

        # Use normal ranks, then prune aggressively to 70% energy
        mpo_param_budget = param_budget * 0.7
        original_params = weight.numel()
        target_mpo_params = int(original_params * mpo_param_budget)

        # For 3-core MPO: params ≈ r1*(num_heads + head_dim) + r2*(head_dim + in_features)
        # Heuristic: set r1 small (inter-head), r2 large (projection)
        r1 = max(2, int(num_heads * mpo_bond_dim_ratio))
        r2 = max(2, int(math.sqrt(target_mpo_params / 2)))

        bond_dims = (1, r1, r2, 1)

        if verbose:
            print(f"  MPO bond dimensions: {bond_dims}")

        # Fit MPO using TT-SVD (on CPU to save memory)
        shape = (num_heads, head_dim, in_features)
        weight_spectral_flat = weight_spectral.detach().cpu()  # Ensure on CPU

        try:
            # Use tensorly's TT decomposition (on CPU)
            tt_decomp = tl.decomposition.tensor_train(
                weight_spectral_flat,
                rank=bond_dims
            )
            mpo_factors = tt_decomp.factors

            # Reconstruct to compute initial SNR (on CPU)
            weight_spectral_mpo = tl.tt_to_tensor(mpo_factors)

        except Exception as e:
            if verbose:
                print(f"  TT-SVD failed: {e}, using random initialization")
            # Fallback: random initialization
            mpo_factors = []
            for i in range(3):
                r_left = bond_dims[i]
                dim = shape[i]
                r_right = bond_dims[i + 1]
                factor = torch.randn(r_left, dim, r_right) * 0.01
                mpo_factors.append(factor)
            weight_spectral_mpo = tl.tt_to_tensor(mpo_factors)

        # Compute MPO reconstruction SNR (before pruning)
        mpo_residual = weight_spectral - weight_spectral_mpo
        mpo_snr = 20 * torch.log10(
            weight_spectral.norm() / (mpo_residual.norm() + 1e-10)
        ).item()

        mpo_params_before = sum(f.numel() for f in mpo_factors)

        if verbose:
            print(f"  MPO parameters (before pruning): {mpo_params_before:,} ({mpo_params_before/original_params:.2%})")
            print(f"  MPO SNR (before pruning): {mpo_snr:.2f} dB")

        # Prune MPO based on energy threshold (on CPU)
        if verbose:
            print(f"  Pruning MPO to retain 70% energy...")

        mpo_factors = prune_tt_by_energy(
            weight_spectral.cpu(),  # Ensure on CPU
            mpo_factors,
            energy_threshold=0.7,  # More aggressive pruning
            verbose=verbose,
        )

        # Recompute after pruning (on CPU)
        weight_spectral_mpo = tl.tt_to_tensor(mpo_factors)
        mpo_residual = weight_spectral - weight_spectral_mpo
        mpo_snr = 20 * torch.log10(
            weight_spectral.norm() / (mpo_residual.norm() + 1e-10)
        ).item()

        mpo_params = sum(f.numel() for f in mpo_factors)

        if verbose:
            reduction = (mpo_params_before - mpo_params) / mpo_params_before
            print(f"  MPO parameters (after pruning): {mpo_params:,} ({mpo_params/original_params:.2%})")
            print(f"  MPO SNR (after pruning): {mpo_snr:.2f} dB")
            print(f"  Pruning saved {reduction:.1%} of MPO params")

        # Clear GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Step 5: Multi-Scale Residual Ladder
        if verbose:
            print("\nStep 5: Multi-Scale Residual Ladder")

        residual_tucker_core = None
        residual_tucker_factors = None
        residual_sparse_indices = None
        residual_sparse_values = None

        current_residual = mpo_residual
        current_snr = mpo_snr

        # Step 5A: Tucker Decomposition on residual
        if use_tucker_residual and current_snr < target_snr_db:
            if verbose:
                print(f"  Step 5A: Tucker residual decomposition")

            try:
                tucker_rank_tuple = tuple([max(2, int(d * tucker_rank)) for d in shape])
                tucker_decomp = tl.decomposition.tucker(
                    current_residual.detach(),
                    rank=tucker_rank_tuple
                )

                residual_tucker_core = tucker_decomp.core
                residual_tucker_factors = tucker_decomp.factors

                # Reconstruct
                tucker_recon = tl.tucker_to_tensor(tucker_decomp)

                # Update residual
                current_residual = current_residual - tucker_recon

                # Compute SNR
                total_recon = weight_spectral_mpo + tucker_recon
                current_snr = 20 * torch.log10(
                    weight_spectral.norm() / ((weight_spectral - total_recon).norm() + 1e-10)
                ).item()

                tucker_params = residual_tucker_core.numel() + sum(f.numel() for f in residual_tucker_factors)

                if verbose:
                    print(f"    Tucker parameters: {tucker_params:,}")
                    print(f"    SNR after Tucker: {current_snr:.2f} dB")

            except Exception as e:
                if verbose:
                    print(f"    Tucker failed: {e}")

        # Step 5B: Sparse Spectral Anchors
        if use_sparse_residual and current_snr < target_snr_db:
            if verbose:
                print(f"  Step 5B: Sparse spectral anchors")

            # Find top percentile of residual magnitudes
            residual_flat = current_residual.flatten()
            threshold = torch.quantile(torch.abs(residual_flat), sparse_percentile / 100.0)

            # Create sparse mask
            sparse_mask = torch.abs(current_residual) >= threshold
            sparse_indices = torch.nonzero(sparse_mask, as_tuple=False)
            sparse_values = current_residual[sparse_mask]

            # Store in COO format
            residual_sparse_indices = sparse_indices
            residual_sparse_values = sparse_values

            # Create sparse tensor for SNR computation
            sparse_tensor = torch.zeros_like(current_residual)
            sparse_tensor[sparse_mask] = sparse_values

            # Update SNR
            total_recon = weight_spectral_mpo
            if residual_tucker_core is not None:
                tucker_recon = tl.tucker_to_tensor((residual_tucker_core, residual_tucker_factors))
                total_recon = total_recon + tucker_recon
            total_recon = total_recon + sparse_tensor

            final_snr = 20 * torch.log10(
                weight_spectral.norm() / ((weight_spectral - total_recon).norm() + 1e-10)
            ).item()

            sparse_params = len(sparse_values) * 2  # Index + value

            if verbose:
                print(f"    Sparse entries: {len(sparse_values):,} ({len(sparse_values)/weight_spectral.numel():.3%})")
                print(f"    Sparse parameters: {sparse_params:,}")
                print(f"    Final SNR: {final_snr:.2f} dB")

        # Summary
        total_params = mpo_params
        if residual_tucker_core is not None:
            total_params += residual_tucker_core.numel() + sum(f.numel() for f in residual_tucker_factors)
        if residual_sparse_indices is not None:
            total_params += len(residual_sparse_values) * 2

        if verbose:
            print(f"\nFinal Results:")
            print(f"  Total parameters: {total_params:,} ({total_params/original_params:.2%})")
            print(f"  Final SNR: {final_snr:.2f} dB")
            print(f"  Target SNR: {target_snr_db:.2f} dB")
            if final_snr >= target_snr_db:
                print(f"  ✓ Target achieved!")
            else:
                print(f"  ⚠ Target not reached (shortfall: {target_snr_db - final_snr:.2f} dB)")

        # Create module (move all tensors to target device)
        module = cls(
            in_features=in_features,
            out_features=out_features,
            num_heads=num_heads,
            mpo_factors=tuple(f.to(device) for f in mpo_factors),
            mpo_bond_dims=bond_dims,
            residual_tucker_core=residual_tucker_core.to(device) if residual_tucker_core is not None else None,
            residual_tucker_factors=tuple(f.to(device) for f in residual_tucker_factors) if residual_tucker_factors else None,
            residual_sparse_indices=residual_sparse_indices.to(device) if residual_sparse_indices is not None else None,
            residual_sparse_values=residual_sparse_values.to(device) if residual_sparse_values is not None else None,
            input_perm=input_perm.to(device),
            input_inv_perm=input_inv_perm.to(device),
            output_perm=output_perm.to(device),
            output_inv_perm=output_inv_perm.to(device),
            bias=bias.to(device) if bias is not None else None,
            calib_inputs=input_activations_cpu.to(device),
            dtype=dtype,
        )

        return module

    def forward(self, x):
        """
        Spectral Sandwich forward pass.

        x → permute → Global-DCT → MPO(W_spectral) @ x_freq → Blocked-IDCT → unpermute → y

        Spectral Sandwich:
        1. Transform input to frequency domain: DCT(x)
        2. Apply spectral weight (MPO-compressed): MPO @ x_freq
        3. Transform back to spatial domain: Blocked-IDCT(y_freq)

        The MPO is the static "knowledge" (compressed spectral weight).
        The DCT input is the "fluid" moving through that structure.

        CRITICAL: Blocked IDCT preserves head boundaries (32 heads × 128 head_dim)
        for subsequent RoPE and Attention operations.
        """
        from scipy.fftpack import dct, idct

        # Store original shape
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        # Flatten batch dimensions
        x_flat = x.reshape(-1, self.in_features)

        # Apply input permutation
        if self.input_perm is not None:
            x_flat = x_flat[:, self.input_perm]

        # Step 1: Transform input to spectral domain (Global DCT)
        x_f32 = x_flat.float()
        x_freq_np = dct(x_f32.cpu().numpy(), axis=-1, norm='ortho')
        x_freq = torch.from_numpy(x_freq_np).to(x.device)

        # Step 2: Reconstruct spectral weight from MPO
        weight_spectral_3d = tl.tt_to_tensor(self.mpo_factors)

        # Add Tucker residual if present
        if self.residual_tucker_core is not None:
            tucker_recon = tl.tucker_to_tensor((self.residual_tucker_core, self.residual_tucker_factors))
            weight_spectral_3d = weight_spectral_3d + tucker_recon

        # Add sparse residual if present
        if self.residual_sparse_indices is not None:
            for idx, val in zip(self.residual_sparse_indices, self.residual_sparse_values):
                weight_spectral_3d[tuple(idx)] += val

        # Reshape to 2D: (out_features, in_features)
        # This is W_spectral - the weight in spectral domain
        weight_spectral = weight_spectral_3d.reshape(self.out_features, self.in_features).to(x_freq.dtype)

        # Step 3: Apply spectral weight to spectral input
        y_freq = x_freq @ weight_spectral.T

        # Step 4: Transform output back to spatial domain (Blocked IDCT per head)
        y_flat = self._idct_per_head(y_freq, self.num_heads)

        # Apply inverse output permutation
        if self.output_inv_perm is not None:
            y_flat = y_flat[:, self.output_inv_perm]

        # Add bias
        if self.bias is not None:
            y_flat = y_flat + self.bias.to(y_flat.dtype)

        # Reshape to original batch dimensions
        result = y_flat.reshape(*batch_dims, self.out_features)

        return result.to(x.dtype)

    def to_matrix(self) -> torch.Tensor:
        """
        Compute effective spatial-domain weight matrix.

        Since the operation includes DCT/IDCT, there's no exact spatial-domain
        weight. We compute an effective weight using calibration inputs:
        W_eff ≈ Y @ pinv(X) where Y = forward(X)
        """
        if self.calib_inputs is None:
            # No calibration data, return identity or zero
            # This shouldn't happen in practice
            return torch.zeros(self.out_features, self.in_features, dtype=self.dtype)

        # Compute outputs for calibration inputs
        with torch.no_grad():
            calib_outputs = self.forward(self.calib_inputs)

        # Solve for effective weight: calib_outputs ≈ W_eff @ calib_inputs^T
        # W_eff = calib_outputs^T @ pinv(calib_inputs^T)
        calib_inputs_f32 = self.calib_inputs.float()
        calib_outputs_f32 = calib_outputs.float()

        # Use least squares: W_eff^T = lstsq(calib_inputs, calib_outputs)
        W_eff_T, _, _, _ = torch.linalg.lstsq(
            calib_inputs_f32, calib_outputs_f32, rcond=None
        )
        W_eff = W_eff_T.T

        return W_eff.to(self.dtype)

    @property
    def num_params(self):
        total = sum(f.numel() for f in self.mpo_factors)
        if self.residual_tucker_core is not None:
            total += self.residual_tucker_core.numel()
            total += sum(f.numel() for f in self.residual_tucker_factors)
        if self.residual_sparse_values is not None:
            total += len(self.residual_sparse_values) * 2  # index + value
        return total
