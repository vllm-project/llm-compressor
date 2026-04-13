"""Efficiency-Driven Multi-Scale Decomposition.

Uses Compression Efficiency = ΔSNR / ΔEffective_Rank to choose when to switch
techniques. Each technique is a "climber" on the SNR/ER curve:

    Fast climbers (high efficiency):
        - ASVD: Dominates at low SNR. Captures the "active logic" — the modes
          that actually see current through activations. Gets massive functional
          SNR gain with small ER. Use until the knee.
        - Adaptive Sparse: Dominates at high SNR. After ASVD saturates, sparse
          targets specific "error spikes" that low-rank can't capture efficiently.

    Slow climbers (low efficiency, optional):
        - Tucker, BlockTT, BlockDiag+LR, Kronecker: Available if budget remains
          after ASVD + Sparse.

Pipeline:
    Phase 1: Efficiency-driven cascade
        Step 1: ASVD rank sweep — grow rank until efficiency knee
        Step 2: Adaptive Sparse — target residual error spikes with remaining budget
        Step 3: Other techniques (if enabled, budget remains, SNR not met)
    Phase 2: Iterative refinement (refit each stage to its ideal target)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from llmcompressor.modifiers.experimental.tucker_linear import TuckerLinear
from llmcompressor.modifiers.experimental.kronecker_linear import KroneckerLinear
from llmcompressor.modifiers.experimental.block_tensorized_linear import (
    BlockTensorizedLinear,
)
from llmcompressor.modifiers.experimental.blockdiag_lowrank_linear import (
    BlockDiagonalLowRankLinear,
)
from llmcompressor.modifiers.experimental.spectral_mpo_linear import SpectralMPOLinear
from llmcompressor.modifiers.experimental.adtn_linear import ColumnSparseLinear


class WeightOnlyLinear(nn.Module):
    """Stores a reconstructed weight matrix for stages like SpectralMPO."""

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

    def to_matrix(self):
        return self.weight

    @property
    def num_params(self):
        return self.weight.numel()


class LowRankLinear(nn.Module):
    """Low-rank linear layer: Y = X @ V @ U^T."""

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.U = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(in_features, rank) * 0.01)

    @classmethod
    def from_svd(cls, weight: torch.Tensor, rank: int):
        """Create low-rank approximation from plain SVD."""
        device = weight.device

        weight_cpu = weight.float().cpu()
        U_full, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)

        U_r = U_full[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]

        out_features, in_features = weight.shape
        layer = cls(in_features, out_features, rank)

        layer.U.data = (U_r @ torch.diag(S_r)).to(dtype=torch.float32, device=device)
        layer.V.data = Vh_r.T.to(dtype=torch.float32, device=device)

        return layer

    @classmethod
    def from_linear_asvd(cls, linear: nn.Linear, input_activations: torch.Tensor,
                         rank: int):
        """Activation-weighted SVD: minimize E[||Wx - W_r x||^2].

        Instead of plain SVD(W), computes SVD(W @ diag(s)) where s_j is the
        RMS activation of channel j. This optimizes for reconstruction quality
        weighted by actual activation statistics.
        """
        W = linear.weight.data.float().cpu()
        X = input_activations.float().cpu()
        device = linear.weight.device
        out_features, in_features = W.shape

        # Channel importance: RMS of activations per input channel
        channel_scale = torch.sqrt((X ** 2).mean(dim=0) + 1e-10)

        # Scale weight columns by activation importance
        # W_scaled = W @ diag(channel_scale)
        W_scaled = W * channel_scale.unsqueeze(0)

        # SVD in activation-weighted space
        U_full, S, Vh = torch.linalg.svd(W_scaled, full_matrices=False)

        # Truncate to rank r
        U_r = U_full[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]  # (rank, in_features)

        # Recover W_r = U_r @ diag(S_r) @ Vh_r @ diag(1/channel_scale)
        # Store as: U_data = U_r @ diag(S_r), V_data = diag(1/s) @ V_r
        # Clamp inverse scale to avoid amplifying near-zero channels
        U_data = U_r @ torch.diag(S_r)  # (out, rank)
        V_r = Vh_r.T  # (in, rank)
        scale_inv = 1.0 / torch.clamp(channel_scale, min=channel_scale.max() * 1e-4)
        V_data = V_r * scale_inv.unsqueeze(1)  # undo scaling per channel

        layer = cls(in_features, out_features, rank)
        layer.U.data = U_data.to(dtype=torch.float32, device=device)
        layer.V.data = V_data.to(dtype=torch.float32, device=device)

        return layer

    @classmethod
    def from_linear_covariance(cls, linear: nn.Linear, input_activations: torch.Tensor,
                               rank: int, eps: float = 1e-6):
        """Covariance-aware SVD: minimize E[||Wx - W_r x||^2] using full covariance.

        Instead of diagonal scaling (ASVD), uses the full activation covariance
        Σ = E[xxᵀ] to find the optimal rank-r approximation:
            SVD(W @ Σ^{1/2}) → absorb Σ^{-1/2} into the right factor.

        Same parameter count as ASVD, but captures off-diagonal activation
        correlations that diagonal scaling misses.

        Args:
            linear: Original linear layer
            input_activations: Calibration activations (n_samples, in_features)
            rank: Target rank
            eps: Noise floor for Σ^{-1/2} to prevent division by zero
        """
        W = linear.weight.data.float().cpu()
        X = input_activations.float().cpu()
        device = linear.weight.device
        out_features, in_features = W.shape

        # Compute activation covariance: Σ = (1/n) XᵀX
        cov = (X.T @ X) / X.shape[0]

        # Eigendecompose: Σ = V Λ Vᵀ (ascending order from eigh)
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)
        except torch._C._LinAlgError:
            import numpy as np
            eigvals_np, eigvecs_np = np.linalg.eigh(cov.numpy())
            eigvals = torch.from_numpy(eigvals_np.copy())
            eigvecs = torch.from_numpy(eigvecs_np.copy())

        # Clamp eigenvalues to non-negative (numerical noise)
        eigvals = eigvals.clamp(min=0)

        # Σ^{1/2} = V @ diag(√λ) @ Vᵀ
        sqrt_eigvals = eigvals.sqrt()
        # W @ Σ^{1/2} = W @ V @ diag(√λ) @ Vᵀ
        W_cov = W @ eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T

        # SVD of W @ Σ^{1/2}
        try:
            U_full, S, Vh = torch.linalg.svd(W_cov, full_matrices=False)
        except torch._C._LinAlgError:
            import numpy as np
            U_np, S_np, Vh_np = np.linalg.svd(W_cov.numpy(), full_matrices=False)
            U_full = torch.from_numpy(U_np.copy())
            S = torch.from_numpy(S_np.copy())
            Vh = torch.from_numpy(Vh_np.copy())

        # Truncate to rank r
        U_r = U_full[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]  # (rank, in_features)

        # W_approx = U_r @ diag(S_r) @ Vh_r @ Σ^{-1/2}
        # Σ^{-1/2} = V @ diag(1/√(λ + ε)) @ Vᵀ
        inv_sqrt_eigvals = 1.0 / (sqrt_eigvals + eps)
        cov_inv_sqrt = eigvecs @ torch.diag(inv_sqrt_eigvals) @ eigvecs.T

        # Absorb Σ^{-1/2} into right factor:
        #   factor_out = U_r @ diag(S_r)          → (out_features, rank)
        #   factor_in  = Vh_r @ Σ^{-1/2}          → (rank, in_features)
        #   V_data     = (Vh_r @ Σ^{-1/2})ᵀ       → (in_features, rank)
        U_data = U_r @ torch.diag(S_r)
        V_data = (Vh_r @ cov_inv_sqrt).T  # (in_features, rank)

        layer = cls(in_features, out_features, rank)
        layer.U.data = U_data.to(dtype=torch.float32, device=device)
        layer.V.data = V_data.to(dtype=torch.float32, device=device)

        return layer

    @classmethod
    def refit(cls, target_linear, input_activations, param_budget, **kwargs):
        """Refit low-rank layer to target param budget."""
        in_features = target_linear.in_features
        out_features = target_linear.out_features
        rank = max(1, param_budget // (in_features + out_features))
        return cls.from_linear_covariance(target_linear, input_activations, rank)

    def forward(self, x):
        V = self.V.to(x.dtype)
        U = self.U.to(x.dtype)
        return x @ V @ U.T

    def to_matrix(self):
        """Reconstruct full weight matrix."""
        return (self.U.data @ self.V.data.T).float()

    @property
    def num_params(self):
        return self.rank * (self.in_features + self.out_features)


def _compute_snr(original, approx):
    """Compute SNR in dB between original and approximation.

    Returns:
        snr_db: SNR in dB
        signal_power_db: signal power in dB (10*log10(var(original)))
        noise_power_db: noise power in dB (10*log10(MSE))
    """
    signal_power = torch.var(original)
    mse = torch.mean((original - approx) ** 2)
    snr_linear = signal_power / (mse + 1e-10)
    snr_db = 10 * torch.log10(snr_linear).item()
    signal_db = 10 * torch.log10(signal_power + 1e-10).item()
    noise_db = 10 * torch.log10(mse + 1e-10).item()
    return snr_db, signal_db, noise_db


def _effective_rank(singular_values):
    """Effective rank via Shannon entropy of normalized SV energy.

    p_i = sigma_i^2 / sum(sigma_j^2)
    H = -sum(p_i * ln(p_i))
    ER = exp(H)
    """
    sv2 = singular_values.float() ** 2
    sv2 = sv2[sv2 > 0]
    if len(sv2) == 0:
        return 0.0
    p = sv2 / sv2.sum()
    H = -(p * p.log()).sum().item()
    return math.exp(H)


def _residual_er(residual_weight, act_norms):
    """Effective rank of activation-weighted residual.

    Measures how structured/compressible the remaining error is.
    Low ER = concentrated in few modes = easy to compress further.
    High ER = spread across many modes = hard to compress further.
    """
    W_cpu = residual_weight.float().cpu()
    norms_cpu = act_norms.float().cpu()
    W_scaled = W_cpu * norms_cpu.unsqueeze(0)
    _, S, _ = torch.linalg.svd(W_scaled, full_matrices=False)
    return _effective_rank(S)


def _make_temp_linear(weight, dtype, in_features, out_features):
    """Create a temporary nn.Linear from a weight tensor."""
    temp = nn.Linear(in_features, out_features, bias=False)
    temp.weight.data = weight.to(dtype)
    return temp


def _detect_num_heads(out_features):
    """Detect number of attention heads from output dimension."""
    for head_dim in [128, 64, 96, 80, 256, 120]:
        if out_features % head_dim == 0:
            return out_features // head_dim
    return 1


def _stage_weight(stage, out_features, in_features, device):
    """Get weight matrix from a stage, handling different stage types."""
    if hasattr(stage, "to_matrix"):
        return stage.to_matrix().float().to(device)
    else:
        return torch.zeros(out_features, in_features, device=device)


def _refit_stage(stage_type, target_linear, input_activations, param_budget,
                 config, device, dtype, current_stage=None):
    """Dispatch to the appropriate technique's refit method.

    Returns (new_stage, new_weight) or (None, None) on failure.
    """
    try:
        if stage_type == "asvd":
            new_stage = LowRankLinear.refit(
                target_linear, input_activations, param_budget,
            )
        elif stage_type == "tucker":
            new_stage = TuckerLinear.refit(
                target_linear, input_activations, param_budget,
                num_modes=config.get("num_modes", 3),
            )
        elif stage_type == "blocktt":
            new_stage = BlockTensorizedLinear.refit(
                target_linear, input_activations, param_budget,
                block_size=config["block_size"],
                num_cores=config["num_cores"],
                current_stage=current_stage,
            )
        elif stage_type == "kronecker":
            new_stage = KroneckerLinear.refit(
                target_linear, input_activations, param_budget,
                factor_size=config.get("factor_size"),
            )
        elif stage_type == "blockdiag_lr":
            new_stage = BlockDiagonalLowRankLinear.refit(
                target_linear, input_activations, param_budget,
                num_blocks=config["num_blocks"],
            )
        elif stage_type in ("spectral_mpo_weight", "spectral_mpo_activation"):
            method = "weight" if "weight" in stage_type else "activation"
            smpo = SpectralMPOLinear.from_linear(
                target_linear,
                num_heads=config["num_heads"],
                input_activations=input_activations,
                param_budget=config.get("param_budget", 0.002),
                block_size=config.get("block_size", 512),
                num_cores=config.get("num_cores", 3),
                rank=config.get("rank", 0.5),
                method=method,
                verbose=False,
            )
            smpo = smpo.to(device)
            original_params = target_linear.weight.numel()
            if smpo.num_params < original_params:
                new_stage = smpo
            else:
                return None, None
        elif stage_type == "sparse":
            new_stage = ColumnSparseLinear.refit(
                target_linear, input_activations.cpu(), param_budget,
                k_cols_per_iter=config.get("k_cols_per_iter", 32),
            )
        elif stage_type == "sparse_adaptive":
            new_stage = ColumnSparseLinear.refit_adaptive(
                target_linear, input_activations.cpu(), param_budget,
                reference_rank_frac=config.get("reference_rank_frac", 0.30),
            )
        else:
            return None, None

        new_stage = new_stage.to(device)
        out_features = target_linear.out_features
        in_features = target_linear.in_features
        new_weight = _stage_weight(new_stage, out_features, in_features, device)
        return new_stage, new_weight
    except Exception as e:
        return None, None


def _compute_stage_budgets(stages, stage_types, stage_weights, W,
                           original_output, input_activations, dtype,
                           total_param_budget):
    """Compute per-stage param budgets based on SNR-efficiency.

    Stages with higher SNR-per-parameter get proportionally more budget.
    Budgets are clamped to [0.5x, 2.0x] of current params.
    """
    num_stages = len(stages)
    total_weight = sum(stage_weights)

    with torch.no_grad():
        output_full = F.linear(input_activations.to(dtype), total_weight.to(dtype)).float()
        snr_full, _, _ = _compute_snr(original_output, output_full)

    marginal_snrs = []
    for i in range(num_stages):
        with torch.no_grad():
            without_i = total_weight - stage_weights[i]
            output_without = F.linear(input_activations.to(dtype), without_i.to(dtype)).float()
            snr_without, _, _ = _compute_snr(original_output, output_without)
        marginal_snrs.append(max(snr_full - snr_without, 0.01))

    efficiencies = [
        marginal_snrs[i] / max(stages[i].num_params, 1)
        for i in range(num_stages)
    ]
    total_efficiency = sum(efficiencies)

    budgets = []
    for i in range(num_stages):
        raw_budget = (efficiencies[i] / total_efficiency) * total_param_budget
        current_params = stages[i].num_params
        # Clamp to [0.5x, 2.0x] of current, but never exceed total budget
        clamped = max(
            int(current_params * 0.5),
            min(int(current_params * 2.0), int(raw_budget)),
        )
        budgets.append(clamped)

    # Normalize so sum(budgets) <= total_param_budget
    budget_sum = sum(budgets)
    if budget_sum > total_param_budget:
        scale = total_param_budget / budget_sum
        budgets = [int(b * scale) for b in budgets]

    return budgets


class GreedyMultiScaleLinear(nn.Module):
    """Greedy multi-scale decomposition with iterative refinement.

    Phase 1: Apply each technique once to the running residual.
    Phase 2: Iteratively refit each stage to its ideal target.
    """

    def __init__(self, stages: nn.ModuleList):
        super().__init__()
        self.stages = stages

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        input_activations: Optional[torch.Tensor] = None,
        target_snr_db: float = 30.0,
        max_refinement_iters: int = 3,
        total_param_budget: Optional[float] = None,  # ratio of original params (e.g. 0.45)
        use_asvd: bool = True,
        svd_budget_fraction: float = 0.6,
        tucker_num_modes: int = 3,
        tucker_rank: float = 0.3,
        spectral_mpo_param_budget: float = 0.002,
        spectral_mpo_block_size: int = 512,
        spectral_mpo_num_cores: int = 3,
        spectral_mpo_rank: float = 0.5,
        # Legacy (ignored)
        spectral_mpo_bond_dim_ratio: float = 0.008,
        spectral_mpo_block_heads: int = 0,
        kronecker_factor_size: Optional[int] = None,
        blocktt_block_size: int = 512,
        blocktt_num_cores: int = 3,
        blocktt_rank: float = 0.5,
        blockdiag_num_blocks: int = 16,
        blockdiag_rank: int = 64,
        sparse_sparsity: float = 0.7,
        use_spectral_mpo: bool = True,
        use_kronecker: bool = True,
        use_blocktt: bool = True,
        use_sparse: bool = True,
        layer_name: Optional[str] = None,
        verbose: bool = True,
    ):
        """Build cascade with efficiency-driven Phase 1 + iterative Phase 2.

        Phase 1 uses Compression Efficiency = ΔSNR / ΔRank to find the optimal
        switch point between ASVD (fast at low SNR) and Sparse (fast at high SNR).

        Args:
            linear: Original linear layer to approximate
            input_activations: Calibration activations (num_samples, in_features)
            target_snr_db: Target SNR in dB
            max_refinement_iters: Max Phase 2 refinement iterations
            total_param_budget: Max params as ratio of original (e.g. 0.45 = 45%)
            layer_name: Layer name for targeted compression (q_proj detection)
            verbose: Print progress
        """
        device = linear.weight.device
        dtype = linear.weight.dtype

        W = linear.weight.data.float().to(device)
        in_features = linear.in_features
        out_features = linear.out_features
        original_params = W.numel()
        min_dim = min(in_features, out_features)

        if input_activations is None:
            input_activations = torch.randn(256, in_features, device=device) * 0.02
        input_activations = input_activations.float().to(device)

        with torch.no_grad():
            original_output = linear(input_activations.to(dtype)).float()

        budget_ratio = total_param_budget if total_param_budget else 0.5
        abs_param_budget = int(budget_ratio * original_params)

        # Efficiency-driven config
        asvd_rank_step = 32
        # Stop ASVD when marginal efficiency drops to this fraction of peak
        efficiency_knee_fraction = 0.1

        if verbose:
            print(f"\nEfficiency-Driven Multi-Scale Decomposition:")
            print(f"  Target SNR: {target_snr_db:.1f} dB")
            print(f"  Param budget: {budget_ratio:.0%} of {original_params:,} = {abs_param_budget:,}")
            print(f"  ASVD rank step: {asvd_rank_step}, knee fraction: {efficiency_knee_fraction}")
            print(f"  Max refinement iters: {max_refinement_iters}\n")

        # ================================================================
        # Phase 1: Efficiency-Driven Cascade
        # ================================================================
        if verbose:
            print("=" * 60)
            print("Phase 1: Efficiency-Driven Cascade")
            print("=" * 60)

        stages = nn.ModuleList()
        stage_types = []
        stage_configs = []
        stage_weights = []  # cached to_matrix() results

        current_weight_approx = torch.zeros_like(W)

        # Activation channel scale (RMS per channel) — used for ASVD and residual ER
        input_acts_cpu = input_activations.float().cpu()
        channel_scale = torch.sqrt((input_acts_cpu ** 2).mean(dim=0) + 1e-10)

        initial_er = _residual_er(W, channel_scale)

        # Covariance analysis: compare diagonal ASVD vs full covariance rotation
        W_cpu = W.float().cpu()
        cov = (input_acts_cpu.T @ input_acts_cpu) / input_acts_cpu.shape[0]
        try:
            cov_eigvals, cov_eigvecs = torch.linalg.eigh(cov)
        except torch._C._LinAlgError:
            import numpy as np
            eigvals_np, eigvecs_np = np.linalg.eigh(cov.numpy())
            cov_eigvals = torch.from_numpy(eigvals_np)
            cov_eigvecs = torch.from_numpy(eigvecs_np)
        # eigh returns ascending order; reverse to descending
        cov_eigvals = cov_eigvals.flip(0)
        cov_eigvecs = cov_eigvecs.flip(1)
        cov_eigvals = cov_eigvals.clamp(min=0)
        cov_er = _effective_rank(cov_eigvals.sqrt())

        # Full covariance-weighted matrix: W @ Σ^{1/2}
        # Σ^{1/2} = V @ diag(sqrt(λ)) @ V^T
        sqrt_eigvals = cov_eigvals.sqrt()
        cov_sqrt = cov_eigvecs @ torch.diag(sqrt_eigvals) @ cov_eigvecs.T
        W_cov = W_cpu @ cov_sqrt

        try:
            _, S_cov, _ = torch.linalg.svd(W_cov, full_matrices=False)
        except torch._C._LinAlgError:
            import numpy as np
            _, S_np, _ = np.linalg.svd(W_cov.numpy(), full_matrices=False)
            S_cov = torch.from_numpy(S_np)
        cov_weighted_er = _effective_rank(S_cov)

        # ASVD (diagonal) for comparison
        W_diag = W_cpu * channel_scale.unsqueeze(0)
        try:
            _, S_diag, _ = torch.linalg.svd(W_diag, full_matrices=False)
        except torch._C._LinAlgError:
            import numpy as np
            _, S_np, _ = np.linalg.svd(W_diag.numpy(), full_matrices=False)
            S_diag = torch.from_numpy(S_np)
        diag_weighted_er = _effective_rank(S_diag)

        if verbose:
            print(f"\n  Initial residual ER: {initial_er:.0f}/{min(out_features, in_features)}")

            # Covariance eigenspectrum summary
            cov_energy = cov_eigvals
            cov_total = cov_energy.sum()
            cov_cumulative = torch.cumsum(cov_energy, dim=0) / cov_total
            print(f"\n  Activation covariance analysis:")
            print(f"    Covariance ER: {cov_er:.0f}/{in_features}")
            for thresh in [0.50, 0.90, 0.95, 0.99]:
                n = int((cov_cumulative < thresh).sum().item()) + 1
                print(f"    {thresh:.0%} covariance energy: {n} eigenvectors "
                      f"({n/in_features:.1%} of {in_features})")
            print(f"    Weight ER — diagonal ASVD: {diag_weighted_er:.0f}, "
                  f"full covariance: {cov_weighted_er:.0f} "
                  f"({'↓' if cov_weighted_er < diag_weighted_er else '↑'}"
                  f"{abs(cov_weighted_er - diag_weighted_er):.0f})")

        # ----------------------------------------------------------
        # Step 1: ASVD — grow rank until efficiency knee
        # ----------------------------------------------------------
        # Compute full ASVD spectrum once (SVD of activation-weighted weight).
        # Then evaluate SNR at increasing ranks using the SV energy spectrum.
        # This is O(1) per rank evaluation — no matrix multiplies needed.
        if use_asvd:
            if verbose:
                print(f"\n  Step 1: ASVD rank sweep")

            residual = W - current_weight_approx
            residual_cpu = residual.float().cpu()
            residual_scaled = residual_cpu * channel_scale.unsqueeze(0)

            try:
                _, S_asvd, _ = torch.linalg.svd(residual_scaled, full_matrices=False)
            except torch._C._LinAlgError:
                import numpy as np
                _, S_np, _ = np.linalg.svd(residual_scaled.numpy(), full_matrices=False)
                S_asvd = torch.from_numpy(S_np)

            # SV-based SNR at each rank: SNR(r) = 10*log10(E_captured / E_residual)
            sv_sq = S_asvd.float() ** 2
            total_energy = sv_sq.sum()
            cumsum_energy = torch.cumsum(sv_sq, dim=0)
            residual_energy = total_energy - cumsum_energy
            snr_curve = 10 * torch.log10(cumsum_energy / (residual_energy + 1e-10))

            # Also compute covariance-aware SNR curve for comparison
            sv_cov_sq = S_cov.float() ** 2
            total_energy_cov = sv_cov_sq.sum()
            cumsum_energy_cov = torch.cumsum(sv_cov_sq, dim=0)
            residual_energy_cov = total_energy_cov - cumsum_energy_cov
            snr_curve_cov = 10 * torch.log10(
                cumsum_energy_cov / (residual_energy_cov + 1e-10)
            )

            # Sweep ranks using covariance-aware spectrum (matches actual decomposition)
            prev_snr = 0.0
            peak_efficiency = 0.0
            best_rank = asvd_rank_step
            cov_er_full = _effective_rank(S_cov)

            if verbose:
                print(f"    Covariance-aware spectrum: ER={cov_er_full:.0f}/{min_dim}, "
                      f"total energy={total_energy_cov.item():.2e}")
                print(f"    {'rank':>8s}  {'covSNR':>8s}  {'ΔSNR':>7s}  {'eff':>10s}  "
                      f"{'resid ER':>9s}  {'diagSNR':>8s}  {'residual':>12s}  {'captured':>8s}  {'params':>7s}")

                # Fine-grained log at powers of 2 up to 64
                fine_ranks = [r for r in [2, 4, 8, 16, 32, 64] if r < asvd_rank_step and r < min_dim]
                prev_fine = 0.0
                for r in fine_ranks:
                    cov_snr_r = snr_curve_cov[r - 1].item() if r <= len(snr_curve_cov) else 0.0
                    diag_snr_r = snr_curve[r - 1].item()
                    delta = cov_snr_r - prev_fine
                    eff_r = delta / r if r == fine_ranks[0] else delta / (r - (fine_ranks[fine_ranks.index(r) - 1]))
                    resid_er_r = _effective_rank(S_cov[r:]) if r < len(S_cov) else 0.0
                    resid_e = residual_energy_cov[r - 1].item() if r <= len(residual_energy_cov) else 0.0
                    cap_pct = cumsum_energy_cov[r - 1].item() / total_energy_cov.item() * 100 if r <= len(cumsum_energy_cov) else 0.0
                    rp = r * (in_features + out_features)
                    print(f"    rank={r:4d}: covSNR={cov_snr_r:6.1f}dB, "
                          f"Δ={delta:+5.2f}dB, eff={eff_r:.4f} dB/rank, "
                          f"resid_ER={resid_er_r:.0f}, diagSNR={diag_snr_r:6.1f}dB, "
                          f"residual={resid_e:.2e}, "
                          f"captured={cap_pct:.1f}%, "
                          f"params={rp/original_params:.1%}")
                    prev_fine = cov_snr_r

            for r in range(asvd_rank_step, min_dim, asvd_rank_step):
                current_snr = snr_curve_cov[r - 1].item() if r <= len(snr_curve_cov) else 0.0
                delta_snr = current_snr - prev_snr
                efficiency = delta_snr / asvd_rank_step  # dB per rank unit
                peak_efficiency = max(peak_efficiency, efficiency)
                resid_er = _effective_rank(S_cov[r:]) if r < len(S_cov) else 0.0

                rank_params = r * (in_features + out_features)

                if verbose and (r <= asvd_rank_step * 4 or r % (asvd_rank_step * 4) == 0):
                    resid_energy_at_r = residual_energy_cov[r - 1].item() if r <= len(residual_energy_cov) else 0.0
                    energy_captured_pct = cumsum_energy_cov[r - 1].item() / total_energy_cov.item() * 100 if r <= len(cumsum_energy_cov) else 0.0
                    diag_snr_r = snr_curve[r - 1].item()
                    print(f"    rank={r:4d}: covSNR={current_snr:6.1f}dB, "
                          f"Δ={delta_snr:+5.2f}dB, eff={efficiency:.4f} dB/rank, "
                          f"resid_ER={resid_er:.0f}, diagSNR={diag_snr_r:6.1f}dB, "
                          f"residual={resid_energy_at_r:.2e}, "
                          f"captured={energy_captured_pct:.1f}%, "
                          f"params={rank_params/original_params:.1%}")

                best_rank = r
                prev_snr = current_snr

                # Stop: target SNR reached
                if current_snr >= target_snr_db:
                    if verbose:
                        print(f"    → Target SVD-SNR reached at rank={r}")
                    break

                # Stop: efficiency dropped to knee_fraction of peak
                if (peak_efficiency > 0
                        and efficiency < peak_efficiency * efficiency_knee_fraction
                        and r > asvd_rank_step * 3):
                    if verbose:
                        print(f"    → Knee at rank={r}: eff={efficiency:.4f} < "
                              f"{peak_efficiency:.4f} × {efficiency_knee_fraction} = "
                              f"{peak_efficiency * efficiency_knee_fraction:.4f}")
                    break

                # Stop: budget limit (leave at least 20% for sparse)
                if rank_params > abs_param_budget * svd_budget_fraction:
                    if verbose:
                        print(f"    → Budget limit at rank={r} "
                              f"({rank_params/original_params:.1%} > "
                              f"{svd_budget_fraction:.0%} of budget)")
                    break

            # Create ASVD stage at the knee rank
            residual_linear = _make_temp_linear(
                residual, dtype, in_features, out_features,
            )
            asvd_stage = LowRankLinear.from_linear_covariance(
                residual_linear, input_activations=input_activations, rank=best_rank,
            )
            asvd_stage = asvd_stage.to(device)
            asvd_weight = _stage_weight(asvd_stage, out_features, in_features, device)

            stages.append(asvd_stage)
            stage_types.append("asvd")
            stage_configs.append({"rank": best_rank})
            stage_weights.append(asvd_weight)
            current_weight_approx = current_weight_approx + asvd_weight

            # Compute actual activation SNR (not just SV-based estimate)
            with torch.no_grad():
                current_output = F.linear(
                    input_activations.to(dtype), current_weight_approx.to(dtype)
                ).float()
            asvd_snr, asvd_sig_db, asvd_noise_db = _compute_snr(original_output, current_output)
            resid_er = _residual_er(W - current_weight_approx, channel_scale)

            if verbose:
                print(f"    ASVD result: rank={best_rank}, "
                      f"params={asvd_stage.num_params:,} ({asvd_stage.num_params/original_params:.1%}), "
                      f"activation SNR={asvd_snr:.2f}dB, "
                      f"noise={asvd_noise_db:.1f}dB, residual ER={resid_er:.0f}")

        # ----------------------------------------------------------
        # Step 2: Adaptive Sparse — target residual error spikes
        # ----------------------------------------------------------
        # After ASVD saturates at the knee, sparse is the fast climber
        # in the high-SNR regime. It directly targets the specific columns
        # responsible for the largest remaining errors.
        if use_sparse:
            with torch.no_grad():
                current_output = F.linear(
                    input_activations.to(dtype), current_weight_approx.to(dtype)
                ).float()
            current_snr, _, _ = _compute_snr(original_output, current_output)

            remaining_budget = abs_param_budget - sum(s.num_params for s in stages)

            if remaining_budget > 0 and current_snr < target_snr_db:
                # Convert remaining budget to column count
                max_cols = remaining_budget // out_features
                target_sparsity = min(1.0, max_cols / in_features)

                if verbose:
                    print(f"\n  Step 2: Adaptive Sparse (remaining budget={remaining_budget:,}, "
                          f"→ {max_cols} cols = {target_sparsity:.1%})")

                residual_weight = W - current_weight_approx
                residual_linear = _make_temp_linear(
                    residual_weight, dtype, in_features, out_features,
                )
                residual_linear.weight.data = residual_linear.weight.data.cpu()

                try:
                    sparse_stage = ColumnSparseLinear.from_linear_adaptive(
                        residual_linear,
                        input_activations=input_activations.cpu(),
                        target_sparsity=target_sparsity,
                        reference_rank_frac=0.30,
                    )
                    sparse_stage = sparse_stage.to(device)
                    sparse_weight = _stage_weight(
                        sparse_stage, out_features, in_features, device,
                    )

                    stages.append(sparse_stage)
                    stage_types.append("sparse_adaptive")
                    stage_configs.append({
                        "sparsity": target_sparsity,
                        "reference_rank_frac": 0.30,
                    })
                    stage_weights.append(sparse_weight)
                    current_weight_approx = current_weight_approx + sparse_weight

                    with torch.no_grad():
                        current_output = F.linear(
                            input_activations.to(dtype),
                            current_weight_approx.to(dtype),
                        ).float()
                    sparse_snr, sparse_sig_db, sparse_noise_db = _compute_snr(original_output, current_output)
                    resid_er = _residual_er(W - current_weight_approx, channel_scale)

                    if verbose:
                        print(f"    Sparse result: {sparse_stage.num_params:,} params "
                              f"({sparse_stage.num_params/original_params:.1%}), "
                              f"SNR → {sparse_snr:.2f}dB (+{sparse_snr - current_snr:.2f}dB), "
                              f"noise={sparse_noise_db:.1f}dB, residual ER={resid_er:.0f}")
                except Exception as e:
                    if verbose:
                        print(f"    Sparse failed: {str(e)[:80]}")

        # ----------------------------------------------------------
        # Step 3: Other techniques (if enabled and budget remains)
        # ----------------------------------------------------------
        with torch.no_grad():
            current_output = F.linear(
                input_activations.to(dtype), current_weight_approx.to(dtype)
            ).float()
        current_snr, _, _ = _compute_snr(original_output, current_output)
        remaining_budget = abs_param_budget - sum(s.num_params for s in stages)

        def _try_add_stage(name, stage_type, create_fn, config):
            """Try to create and add a stage. Returns True if added."""
            nonlocal current_weight_approx, current_snr

            residual_weight = W - current_weight_approx
            temp_linear = _make_temp_linear(
                residual_weight, dtype, in_features, out_features,
            )

            try:
                stage, weight_matrix = create_fn(temp_linear)
                if stage is None:
                    return False

                stage = stage.to(device)
                weight_matrix = _stage_weight(
                    stage, out_features, in_features, device,
                )

                new_approx = current_weight_approx + weight_matrix
                with torch.no_grad():
                    new_output = F.linear(
                        input_activations.to(dtype), new_approx.to(dtype)
                    ).float()
                new_snr, new_sig_db, new_noise_db = _compute_snr(original_output, new_output)
                delta_snr = new_snr - current_snr

                if delta_snr > 0.01:
                    stages.append(stage)
                    stage_types.append(stage_type)
                    stage_configs.append(config)
                    stage_weights.append(weight_matrix)
                    current_weight_approx = new_approx
                    current_snr = new_snr
                    resid_er = _residual_er(W - current_weight_approx, channel_scale)
                    if verbose:
                        print(f"    {name}: {stage.num_params:,} params, "
                              f"SNR → {new_snr:.2f}dB (+{delta_snr:.2f}), "
                              f"noise={new_noise_db:.1f}dB, residual ER={resid_er:.0f}")
                    return True
                else:
                    if verbose:
                        print(f"    {name}: skipped (ΔSNR={delta_snr:.2f}dB)")
            except Exception as e:
                if verbose:
                    print(f"    {name}: failed ({str(e)[:60]})")
            return False

        if remaining_budget > 0 and current_snr < target_snr_db:
            if verbose:
                print(f"\n  Step 3: Other techniques "
                      f"(remaining={remaining_budget:,} params, SNR={current_snr:.1f}dB)")

            # Tucker
            tucker_config = {"num_modes": tucker_num_modes, "rank": tucker_rank}
            _try_add_stage(
                "Tucker", "tucker",
                lambda tl: (
                    TuckerLinear.from_linear(
                        tl, rank=tucker_rank, num_modes=tucker_num_modes,
                        input_activations=input_activations, verbose=False,
                    ),
                    None,
                ),
                tucker_config,
            )

            # Block Tensor Train
            if use_blocktt:
                blocktt_config = {
                    "block_size": blocktt_block_size,
                    "num_cores": blocktt_num_cores,
                    "rank": blocktt_rank,
                }
                _try_add_stage(
                    "BlockTT", "blocktt",
                    lambda tl: (
                        BlockTensorizedLinear.from_linear(
                            tl, block_size=blocktt_block_size,
                            num_cores=blocktt_num_cores, rank=blocktt_rank,
                            input_activations=input_activations,
                        ),
                        None,
                    ),
                    blocktt_config,
                )

            # Block-Diagonal + Low-Rank
            bdlr_config = {
                "num_blocks": blockdiag_num_blocks,
                "rank": blockdiag_rank,
            }
            _try_add_stage(
                "BlockDiag+LR", "blockdiag_lr",
                lambda tl: (
                    BlockDiagonalLowRankLinear.from_linear(
                        tl, num_blocks=blockdiag_num_blocks,
                        rank=blockdiag_rank, verbose=False,
                    ),
                    None,
                ),
                bdlr_config,
            )

            # Kronecker
            if use_kronecker:
                kron_config = {"factor_size": kronecker_factor_size}
                _try_add_stage(
                    "Kronecker", "kronecker",
                    lambda tl: (
                        KroneckerLinear.from_linear(
                            tl, factor_size=kronecker_factor_size, verbose=False,
                        ),
                        None,
                    ),
                    kron_config,
                )

        # Phase 1 summary
        with torch.no_grad():
            current_output = F.linear(
                input_activations.to(dtype), current_weight_approx.to(dtype)
            ).float()
        phase1_snr, phase1_sig_db, phase1_noise_db = _compute_snr(original_output, current_output)
        phase1_resid_er = _residual_er(W - current_weight_approx, channel_scale)
        phase1_params = sum(s.num_params for s in stages)

        if verbose:
            print(f"\n  Phase 1 Results:")
            print(f"    Stages: {', '.join(stage_types)}")
            print(f"    SNR: {phase1_snr:.2f} dB, noise={phase1_noise_db:.1f}dB, residual ER={phase1_resid_er:.0f}")
            print(f"    Params: {phase1_params:,} / {original_params:,} = {phase1_params/original_params:.2%}")

        # ================================================================
        # Phase 2: Iterative Refinement
        # ================================================================
        if max_refinement_iters > 0 and len(stages) > 0:
            if verbose:
                print(f"\n{'=' * 60}")
                print("Phase 2: Iterative Refinement")
                print("=" * 60)

            # Compute absolute param budget
            if total_param_budget is not None:
                abs_param_budget = int(total_param_budget * W.numel())
            else:
                # Default: keep current total
                abs_param_budget = phase1_params

            for ref_iter in range(max_refinement_iters):
                with torch.no_grad():
                    current_output = F.linear(
                        input_activations.to(dtype), current_weight_approx.to(dtype)
                    ).float()
                current_snr, cur_sig_db, cur_noise_db = _compute_snr(original_output, current_output)
                cur_resid_er = _residual_er(W - current_weight_approx, channel_scale)

                if verbose:
                    total_p = sum(s.num_params for s in stages)
                    print(f"\n  Refinement iter {ref_iter + 1}:")
                    print(f"    Current SNR: {current_snr:.2f} dB, noise={cur_noise_db:.1f}dB, residual ER={cur_resid_er:.0f}, Params: {total_p:,}")

                if current_snr >= target_snr_db:
                    if verbose:
                        print(f"    Target SNR reached!")
                    break

                # Compute per-stage budgets
                budgets = _compute_stage_budgets(
                    stages, stage_types, stage_weights, W,
                    original_output, input_activations, dtype,
                    abs_param_budget,
                )

                improved_any = False

                for i in range(len(stages)):
                    # Compute ideal target for this stage
                    other_weight = sum(
                        stage_weights[j] for j in range(len(stages)) if j != i
                    )
                    ideal_target_weight = W - other_weight

                    target_linear = _make_temp_linear(
                        ideal_target_weight, dtype, in_features, out_features,
                    )

                    new_stage, new_weight = _refit_stage(
                        stage_types[i], target_linear, input_activations,
                        budgets[i], stage_configs[i], device, dtype,
                        current_stage=stages[i],
                    )

                    if new_stage is None:
                        continue

                    # Check if refit improves overall SNR
                    new_total_weight = other_weight + new_weight
                    with torch.no_grad():
                        new_output = F.linear(
                            input_activations.to(dtype), new_total_weight.to(dtype)
                        ).float()
                    new_snr, new_sig_db, new_noise_db = _compute_snr(original_output, new_output)

                    # Check budget: total params after refit must stay within budget
                    new_total_params = (
                        sum(stages[j].num_params for j in range(len(stages)) if j != i)
                        + new_stage.num_params
                    )

                    if new_snr > current_snr and new_total_params <= abs_param_budget:
                        old_params = stages[i].num_params
                        stages[i] = new_stage
                        stage_weights[i] = new_weight
                        current_weight_approx = new_total_weight
                        current_snr = new_snr
                        improved_any = True
                        new_resid_er = _residual_er(W - current_weight_approx, channel_scale)

                        if verbose:
                            print(f"    {stage_types[i]}: {old_params:,} -> {new_stage.num_params:,} params, "
                                  f"SNR -> {new_snr:.2f} dB, noise={new_noise_db:.1f}dB, residual ER={new_resid_er:.0f}")
                    else:
                        reason = ""
                        if new_snr <= current_snr:
                            reason = f"SNR {new_snr:.2f} <= {current_snr:.2f}"
                        else:
                            reason = f"budget {new_total_params:,} > {abs_param_budget:,}"
                        if verbose:
                            print(f"    {stage_types[i]}: refit rejected ({reason})")

                if not improved_any:
                    if verbose:
                        print(f"    No improvements, stopping refinement.")
                    break

        # Final summary
        with torch.no_grad():
            current_output = F.linear(
                input_activations.to(dtype), current_weight_approx.to(dtype)
            ).float()
        final_snr, final_sig_db, final_noise_db = _compute_snr(original_output, current_output)
        final_resid_er = _residual_er(W - current_weight_approx, channel_scale)
        total_params = sum(s.num_params for s in stages)
        original_params = W.numel()

        if verbose:
            print(f"\nFinal Results:")
            print(f"  Stages: {len(stages)}")
            print(f"  Final SNR: {final_snr:.2f} dB, noise={final_noise_db:.1f}dB, residual ER={final_resid_er:.0f}")
            print(f"  Total params: {total_params:,} ({total_params/original_params:.2%})")
            print(f"  Compression: {100*(1-total_params/original_params):.1f}%")

        module = cls(stages)
        module.to(device)
        return module

    def forward(self, x):
        """Sum outputs of all stages."""
        output = 0
        for stage in self.stages:
            output = output + stage(x)
        return output

    @property
    def num_params(self):
        return sum(stage.num_params for stage in self.stages)

    def to_matrix(self):
        """Reconstruct full weight matrix."""
        if not self.stages:
            return None

        first_stage = self.stages[0]
        if hasattr(first_stage, "out_features"):
            out_features = first_stage.out_features
            in_features = first_stage.in_features
        else:
            return None

        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else "cpu"
        total = torch.zeros(out_features, in_features, device=device)

        for stage in self.stages:
            if hasattr(stage, "to_matrix"):
                total = total + stage.to_matrix()

        return total


__all__ = ["GreedyMultiScaleLinear", "LowRankLinear", "WeightOnlyLinear"]
