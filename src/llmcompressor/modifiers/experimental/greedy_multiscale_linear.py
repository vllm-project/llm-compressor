"""Greedy Multi-Scale Decomposition: 7-Component Cascade (q_proj) / 5-Component (others).

Mathematical form (q_proj):
    y = Tucker(x) + SpectralMPO_weight(x) + SpectralMPO_act(x) + BlockTT(x) + Kronecker(x) + BlockDiagLR(x) + Sparse(x) + ...

Mathematical form (other layers):
    y = Tucker(x) + BlockTT(x) + Kronecker(x) + BlockDiagLR(x) + Sparse(x) + ...

Strategy (per stage):
    1. Tucker: Global multi-dimensional structure (dual spectral reordering)
    2A. SpectralMPO (weight): Direct DCT(Weight) - global structure/prior [q_proj only]
    2B. SpectralMPO (activation): LSTSQ - task-aware/evidence [q_proj only]
    3. Block Tensor Train: Block-wise structured tensor decomposition
    4. Kronecker: Repeating/block-periodic patterns (fractal-like, super efficient)
    5. Block-Diagonal + Low-Rank: Local clusters + global communication
    6. Sparse: Outlier features and sharp edges

Geometric perspectives:
    - Tucker: Multi-dimensional correlations with spectral reordering
    - SpectralMPO (weight): Frequency-domain structure from weight matrix (prior knowledge)
    - SpectralMPO (activation): Frequency-domain structure from activations (task-specific evidence)
    - Block TT: Divide matrix into blocks, tensor train per block
    - Kronecker: Repeating patterns (B ⊗ C), parameter-efficient
    - BlockDiag+LR: Dense local clusters + low-rank global communication
    - Sparse: Greedy column selection for outliers

Why this order:
    - Tucker FIRST: Captures global structure (benefits from full signal for spectral reordering)
    - SpectralMPO: Frequency-domain compression (only for q_proj with multi-head structure)
    - Block TT: Structured spatial decomposition
    - Kronecker: Repeating pattern decomposition
    - BlockDiag+LR: Stable local + global decomposition
    - Sparse: Captures outliers/sharp features that structured methods miss

Benefits:
    - Structured methods (Tucker, SpectralMPO, BlockTT, BlockDiag+LR) work on full signal
    - DUAL SpectralMPO for q_proj: Weight-based (prior) + Activation-based (evidence)
    - Sparse handles outliers the structured methods couldn't compress
    - Each component attacks error from different geometric perspective
    - Numerically stable (small components)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from llmcompressor.modifiers.experimental.tucker_linear import TuckerLinear
from llmcompressor.modifiers.experimental.kronecker_linear import KroneckerLinear
from llmcompressor.modifiers.experimental.block_tensorized_linear import BlockTensorizedLinear
from llmcompressor.modifiers.experimental.blockdiag_lowrank_linear import BlockDiagonalLowRankLinear
from llmcompressor.modifiers.experimental.spectral_mpo_linear import SpectralMPOLinear
from llmcompressor.modifiers.experimental.adtn_linear import ColumnSparseLinear


class LowRankLinear(nn.Module):
    """Low-rank linear layer: Y = U @ V^T @ X."""

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # U: (out_features, rank), V: (in_features, rank)
        self.U = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(in_features, rank) * 0.01)

    @classmethod
    def from_svd(cls, weight: torch.Tensor, rank: int):
        """Create low-rank approximation from SVD."""
        device = weight.device
        # Store as float32 for numerical precision, convert to input dtype in forward()
        target_dtype = torch.float32

        # Perform SVD on CPU for numerical stability
        weight_cpu = weight.float().cpu()
        U_full, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)

        # Truncate to rank
        U_r = U_full[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]

        # Create layer
        out_features, in_features = weight.shape
        layer = cls(in_features, out_features, rank)

        # Set parameters: U @ S, V^T in float32, move to correct device
        layer.U.data = (U_r @ torch.diag(S_r)).to(dtype=target_dtype, device=device)
        layer.V.data = Vh_r.T.to(dtype=target_dtype, device=device)

        return layer

    @classmethod
    def from_activations(cls, input_acts: torch.Tensor, target_acts: torch.Tensor, rank: int):
        """Fit low-rank layer to approximate: target_acts ≈ W @ input_acts.

        Args:
            input_acts: (num_samples, in_features)
            target_acts: (num_samples, out_features)
            rank: Rank of low-rank approximation
        """
        # Solve for W such that target_acts ≈ W @ input_acts^T
        # W ≈ target_acts @ input_acts^+ (pseudoinverse)

        # Use SVD for numerical stability
        W = target_acts.T @ torch.linalg.pinv(input_acts.T)

        # Create low-rank approximation of W
        return cls.from_svd(W, rank)

    def forward(self, x):
        """Forward: Y = U @ V^T @ X."""
        # x: (..., in_features)
        # V^T @ x: (..., rank)
        # U @ (V^T @ x): (..., out_features)
        # Ensure dtype matches input
        V = self.V.to(x.dtype)
        U = self.U.to(x.dtype)
        return x @ V @ U.T

    @property
    def num_params(self):
        return self.rank * (self.in_features + self.out_features)


class GreedyMultiScaleLinear(nn.Module):
    """Greedy multi-scale decomposition with cascaded MPO + Low-Rank stages.

    Structure:
        y = MPO_1(x) + LR_1(x) + MPO_2(x) + LR_2(x) + ...

    Each stage captures residual information missed by previous stages.
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
        max_stages: int = 5,
        tucker_num_modes: int = 3,
        tucker_rank: float = 0.3,  # Low rank for small Tucker cores
        spectral_mpo_bond_dim_ratio: float = 0.008,  # Bond dimension ratio for Spectral MPO
        spectral_mpo_param_budget: float = 0.002,  # Parameter budget for Spectral MPO
        kronecker_factor_size: Optional[int] = None,  # Auto: sqrt of dims
        blocktt_block_size: int = 512,  # Block size for Block TT
        blocktt_num_cores: int = 3,  # Number of cores per block
        blocktt_rank: float = 0.5,  # Rank for Block TT
        blockdiag_num_blocks: int = 16,  # Number of diagonal blocks
        blockdiag_rank: int = 64,  # Low-rank component rank
        sparse_sparsity: float = 0.7,  # Column sparsity (keep 30% of columns)
        use_spectral_mpo: bool = True,  # Include Spectral MPO stages (q_proj only)
        use_kronecker: bool = True,  # Include Kronecker stages
        use_blocktt: bool = True,  # Include Block TT stages
        use_sparse: bool = True,  # Include column-sparse stages
        layer_name: Optional[str] = None,  # Layer name for targeted compression
        verbose: bool = True,
    ):
        """Greedily build cascade to reach target SNR.

        Args:
            linear: Original linear layer to approximate
            input_activations: Calibration activations (num_samples, in_features)
            target_snr_db: Target SNR in dB
            max_stages: Maximum number of stages
            tucker_num_modes: Number of modes for Tucker decomposition
            tucker_rank: Rank ratio for Tucker core (low for small components)
            kronecker_factor_size: Factor size for Kronecker (default: sqrt of dims)
            blocktt_block_size: Block size for Block Tensor Train
            blocktt_num_cores: Number of TT cores per block
            blocktt_rank: Rank ratio for Block TT
            blockdiag_num_blocks: Number of diagonal blocks (local clusters)
            blockdiag_rank: Rank of low-rank component (global communication)
            sparse_sparsity: Target sparsity for column-sparse (0.7 = keep 30% columns)
            use_kronecker: Include Kronecker stages (super parameter-efficient)
            use_blocktt: Include Block Tensor Train stages
            use_sparse: Include column-sparse stages in cascade
            verbose: Print progress

        Stage order per iteration:
            1. Tucker (global structure with dual spectral reordering)
            2. Block Tensor Train (block-wise structured decomposition)
            3. Block-Diagonal + Low-Rank (local clusters + global communication)
            4. Sparse (outlier features and sharp edges)
            5. Kronecker (repeating patterns, super efficient)
        """

        def compute_snr(original, approx):
            signal_power = torch.var(original)
            mse = torch.mean((original - approx) ** 2)
            snr_linear = signal_power / (mse + 1e-10)
            return 10 * torch.log10(snr_linear).item()

        stages = nn.ModuleList()

        # Track device and dtype
        device = linear.weight.device
        dtype = linear.weight.dtype

        W = linear.weight.data.float().to(device)
        in_features = linear.in_features
        out_features = linear.out_features

        # Generate activations if not provided
        if input_activations is None:
            input_activations = torch.randn(256, in_features, device=device) * 0.02

        input_activations = input_activations.float().to(device)

        # Compute original outputs
        with torch.no_grad():
            original_output = linear(input_activations.to(dtype)).float()

        # Start with zero approximation
        current_output = torch.zeros_like(original_output)
        current_weight_approx = torch.zeros_like(W)

        if verbose:
            print(f"\nGreedy Multi-Scale Decomposition:")
            print(f"  Target SNR: {target_snr_db:.1f} dB")
            print(f"  Max stages: {max_stages}")
            print(f"  Tucker: num_modes={tucker_num_modes}, rank={tucker_rank}")
            print(f"  SpectralMPO: bond_ratio={spectral_mpo_bond_dim_ratio}, budget={spectral_mpo_param_budget}, variants=['weight','activation'], enabled={use_spectral_mpo}")
            print(f"  Kronecker: factor_size={kronecker_factor_size or 'auto'}, enabled={use_kronecker}")
            print(f"  BlockTT: block_size={blocktt_block_size}, num_cores={blocktt_num_cores}, rank={blocktt_rank}, enabled={use_blocktt}")
            print(f"  Sparse: sparsity={sparse_sparsity}, enabled={use_sparse}")
            print(f"  BlockDiag+LR: num_blocks={blockdiag_num_blocks}, rank={blockdiag_rank}\n")

        for stage_idx in range(max_stages):
            # Compute current SNR
            current_snr = compute_snr(original_output, current_output)

            if verbose:
                print(f"Stage {stage_idx + 1}:")
                print(f"  Current SNR: {current_snr:.2f} dB")

            if current_snr >= target_snr_db:
                if verbose:
                    print(f"  ✓ Target SNR reached!")
                break

            # Step 1: Fit Tucker to current residual (with spectral reordering)
            residual_output = original_output - current_output
            residual_weight = W - current_weight_approx

            # Create temporary linear with residual weights
            temp_linear_tucker = nn.Linear(in_features, out_features, bias=False)
            temp_linear_tucker.weight.data = residual_weight.to(dtype)

            try:
                # Create TuckerLinear (verbose=False to avoid cluttering output)
                tucker = TuckerLinear.from_linear(
                    temp_linear_tucker,
                    rank=tucker_rank,
                    num_modes=tucker_num_modes,
                    input_activations=input_activations,
                    verbose=False,
                )

                # Move Tucker to correct device
                tucker = tucker.to(device)

                with torch.no_grad():
                    tucker_output = tucker(input_activations.to(dtype)).float()

                tucker_weight = tucker.to_matrix().float().to(device)
                tucker_snr = compute_snr(residual_output, tucker_output)

                if verbose:
                    status = ""
                    if tucker_snr > 0.01:  # Only keep if it improves SNR
                        status = " ✓"
                    else:
                        status = " (skipped)"
                    print(f"  Tucker: {tucker.num_params:,} params, SNR {tucker_snr:+.2f} dB{status}")

                # Only add if it improves SNR
                if tucker_snr > 0.01:
                    stages.append(tucker)
                    current_output = current_output + tucker_output
                    current_weight_approx = current_weight_approx + tucker_weight

            except Exception as e:
                if verbose:
                    print(f"  Tucker: failed ({str(e)[:60]})")

            # Step 2A: Spectral MPO (Weight-based) - Prior/Global Structure
            # Only apply to q_proj layers (k_proj/v_proj have different head_dim structure)
            if use_spectral_mpo and layer_name is not None and "q_proj" in layer_name:
                residual_output_3a = original_output - current_output
                residual_weight_3a = W - current_weight_approx

                # Create temporary linear with residual weights
                temp_linear_smpo_weight = nn.Linear(in_features, out_features, bias=False)
                temp_linear_smpo_weight.weight.data = residual_weight_3a.to(dtype)

                try:
                    # For q_proj, detect num_heads from dimensions
                    # Common configs:
                    #   Llama-3.1-8B: 32 heads * 128 head_dim = 4096
                    #   Llama-2-7B: 32 heads * 128 head_dim = 4096
                    #   Smaller models: 8 heads * 64 head_dim = 512
                    num_heads = None
                    for head_dim in [128, 64, 96, 80, 256, 120]:
                        if out_features % head_dim == 0:
                            num_heads = out_features // head_dim
                            if verbose and stage_idx == 0:  # Only print once
                                print(f"  SpectralMPO: Detected {num_heads} heads × {head_dim} head_dim = {out_features}")
                            break

                    if num_heads is None:
                        # Fallback: assume single head
                        num_heads = 1
                        if verbose and stage_idx == 0:
                            print(f"  SpectralMPO: Could not detect num_heads for out_features={out_features}, using 1")

                    spectral_mpo_weight = SpectralMPOLinear.from_linear(
                        temp_linear_smpo_weight,
                        num_heads=num_heads,
                        input_activations=input_activations,
                        target_snr_db=10.0,  # Lower target for single component
                        param_budget=spectral_mpo_param_budget,
                        mpo_bond_dim_ratio=spectral_mpo_bond_dim_ratio,
                        method="weight",  # Weight-based: DCT(W) → Prior
                        use_tucker_residual=False,  # Disable internal Tucker (we have external Tucker)
                        use_sparse_residual=False,  # Disable internal sparse (we have external sparse)
                        verbose=False,
                    )

                    # Move to correct device
                    spectral_mpo_weight = spectral_mpo_weight.to(device)

                    with torch.no_grad():
                        # Use to_matrix() since forward() may not be fully implemented
                        smpo_weight_matrix = spectral_mpo_weight.to_matrix().float().to(device)
                        smpo_weight_output = F.linear(input_activations.to(dtype), smpo_weight_matrix.to(dtype)).float()

                    smpo_weight_snr = compute_snr(residual_output_3a, smpo_weight_output)

                    if verbose:
                        status = ""
                        if smpo_weight_snr > 0.01:
                            status = " ✓"
                        else:
                            status = " (skipped)"
                        print(f"  SpectralMPO(weight): {spectral_mpo_weight.num_params:,} params, SNR {smpo_weight_snr:+.2f} dB{status}")

                    # Only add if it improves SNR
                    if smpo_weight_snr > 0.01:
                        # Store as weight matrix instead of full SpectralMPO to avoid forward() issues
                        # Create a simple wrapper that just stores the reconstructed weight
                        class WeightOnlyLinear(nn.Module):
                            def __init__(self, weight):
                                super().__init__()
                                self.weight = nn.Parameter(weight, requires_grad=False)
                                self.in_features = weight.shape[1]
                                self.out_features = weight.shape[0]
                                self.num_params = weight.numel()  # For compatibility

                            def forward(self, x):
                                return F.linear(x, self.weight.to(x.dtype))

                            def to_matrix(self):
                                return self.weight

                        weight_layer = WeightOnlyLinear(smpo_weight_matrix)
                        stages.append(weight_layer)
                        current_output = current_output + smpo_weight_output
                        current_weight_approx = current_weight_approx + smpo_weight_matrix

                except Exception as e:
                    if verbose:
                        print(f"  SpectralMPO(weight): failed ({str(e)[:60]})")

            # Step 2B: Spectral MPO (Activation-based) - Evidence/Task-Aware
            if use_spectral_mpo and layer_name is not None and "q_proj" in layer_name:
                residual_output_3b = original_output - current_output
                residual_weight_3b = W - current_weight_approx

                # Create temporary linear with residual weights
                temp_linear_smpo_act = nn.Linear(in_features, out_features, bias=False)
                temp_linear_smpo_act.weight.data = residual_weight_3b.to(dtype)

                try:
                    # Use same num_heads as above
                    spectral_mpo_act = SpectralMPOLinear.from_linear(
                        temp_linear_smpo_act,
                        num_heads=num_heads,
                        input_activations=input_activations,
                        target_snr_db=10.0,  # Lower target for single component
                        param_budget=spectral_mpo_param_budget,
                        mpo_bond_dim_ratio=spectral_mpo_bond_dim_ratio,
                        method="activation",  # Activation-based: LSTSQ → Evidence
                        use_tucker_residual=False,  # Disable internal Tucker (we have external Tucker)
                        use_sparse_residual=False,  # Disable internal sparse (we have external sparse)
                        verbose=False,
                    )

                    # Move to correct device
                    spectral_mpo_act = spectral_mpo_act.to(device)

                    with torch.no_grad():
                        # Use to_matrix()
                        smpo_act_matrix = spectral_mpo_act.to_matrix().float().to(device)
                        smpo_act_output = F.linear(input_activations.to(dtype), smpo_act_matrix.to(dtype)).float()

                    smpo_act_snr = compute_snr(residual_output_3b, smpo_act_output)

                    if verbose:
                        status = ""
                        if smpo_act_snr > 0.01:
                            status = " ✓"
                        else:
                            status = " (skipped)"
                        print(f"  SpectralMPO(activation): {spectral_mpo_act.num_params:,} params, SNR {smpo_act_snr:+.2f} dB{status}")

                    # Only add if it improves SNR
                    if smpo_act_snr > 0.01:
                        class WeightOnlyLinear(nn.Module):
                            def __init__(self, weight):
                                super().__init__()
                                self.weight = nn.Parameter(weight, requires_grad=False)
                                self.in_features = weight.shape[1]
                                self.out_features = weight.shape[0]
                                self.num_params = weight.numel()

                            def forward(self, x):
                                return F.linear(x, self.weight.to(x.dtype))

                            def to_matrix(self):
                                return self.weight

                        act_layer = WeightOnlyLinear(smpo_act_matrix)
                        stages.append(act_layer)
                        current_output = current_output + smpo_act_output
                        current_weight_approx = current_weight_approx + smpo_act_matrix

                except Exception as e:
                    if verbose:
                        print(f"  SpectralMPO(activation): failed ({str(e)[:60]})")

            elif use_spectral_mpo and verbose and stage_idx == 0:
                # Log why Spectral MPO was skipped (only once)
                if layer_name is None:
                    print(f"  SpectralMPO: skipped (no layer_name provided)")
                elif "q_proj" not in layer_name:
                    print(f"  SpectralMPO: skipped (only applies to q_proj, got {layer_name})")

            # Step 3: Block Tensor Train for block-wise structured decomposition
            if use_blocktt:
                residual_output_3 = original_output - current_output
                residual_weight_3 = W - current_weight_approx

                # Create temporary linear with residual weights
                temp_linear_btt = nn.Linear(in_features, out_features, bias=False)
                temp_linear_btt.weight.data = residual_weight_3.to(dtype)

                try:
                    # Create BlockTensorizedLinear (verbose=False)
                    blocktt = BlockTensorizedLinear.from_linear(
                        temp_linear_btt,
                        block_size=blocktt_block_size,
                        num_cores=blocktt_num_cores,
                        rank=blocktt_rank,
                        input_activations=input_activations,
                    )

                    # Move to correct device
                    blocktt = blocktt.to(device)

                    with torch.no_grad():
                        blocktt_output = blocktt(input_activations.to(dtype)).float()

                    blocktt_weight = blocktt.to_matrix().float().to(device)
                    blocktt_snr = compute_snr(residual_output_3, blocktt_output)

                    if verbose:
                        status = ""
                        if blocktt_snr > 0.01:
                            status = " ✓"
                        else:
                            status = " (skipped)"
                        print(f"  BlockTT: {blocktt.num_params:,} params, SNR {blocktt_snr:+.2f} dB{status}")

                    # Only add if it improves SNR
                    if blocktt_snr > 0.01:
                        stages.append(blocktt)
                        current_output = current_output + blocktt_output
                        current_weight_approx = current_weight_approx + blocktt_weight

                except Exception as e:
                    if verbose:
                        print(f"  BlockTT: failed ({str(e)[:60]})")

            # Step 5: Fit Block-Diagonal + Low-Rank to remaining residual
            residual_output_5 = original_output - current_output
            residual_weight_5 = W - current_weight_approx

            # Create temporary linear with residual weights
            temp_linear_bdlr = nn.Linear(in_features, out_features, bias=False)
            temp_linear_bdlr.weight.data = residual_weight_5.to(dtype)

            try:
                # Create BlockDiagonalLowRankLinear (verbose=False to avoid cluttering output)
                bdlr = BlockDiagonalLowRankLinear.from_linear(
                    temp_linear_bdlr,
                    num_blocks=blockdiag_num_blocks,
                    rank=blockdiag_rank,
                    verbose=False,
                )

                # Move to correct device
                bdlr = bdlr.to(device)

                with torch.no_grad():
                    bdlr_output = bdlr(input_activations.to(dtype)).float()

                bdlr_weight = bdlr.to_matrix().float().to(device)
                bdlr_snr = compute_snr(residual_output_5, bdlr_output)

                if verbose:
                    status = ""
                    if bdlr_snr > 0.01:
                        status = " ✓"
                    else:
                        status = " (skipped)"
                    print(f"  BlockDiag+LR: {bdlr.num_params:,} params, SNR {bdlr_snr:+.2f} dB{status}")

                # Only add if it improves SNR
                if bdlr_snr > 0.01:
                    stages.append(bdlr)
                    current_output = current_output + bdlr_output
                    current_weight_approx = current_weight_approx + bdlr_weight

            except Exception as e:
                if verbose:
                    print(f"  BlockDiag+LR: failed ({str(e)[:60]})")

            # Step 6: Column-sparse to capture important features/outliers
            if use_sparse:
                residual_output_6 = original_output - current_output
                residual_weight_6 = W - current_weight_approx

                # Create temporary linear with residual weights
                temp_linear_sparse = nn.Linear(in_features, out_features, bias=False)
                temp_linear_sparse.weight.data = residual_weight_6.cpu().to(dtype)

                try:
                    # Create column-sparse layer
                    sparse = ColumnSparseLinear.from_linear(
                        temp_linear_sparse,
                        input_activations=input_activations.cpu(),
                        target_sparsity=sparse_sparsity,
                        k_cols_per_iter=32,  # Greedy column selection
                    )

                    # Move to correct device
                    sparse = sparse.to(device)

                    with torch.no_grad():
                        sparse_output = sparse(input_activations.to(dtype)).float()

                    # Reconstruct sparse weight matrix for tracking
                    # ColumnSparseLinear stores selected_columns and partial weight
                    sparse_weight_full = torch.zeros(out_features, in_features, device=device)
                    sparse_weight_full[:, sparse.selected_columns] = sparse.weight.data.float()

                    sparse_snr = compute_snr(residual_output_6, sparse_output)

                    if verbose:
                        status = ""
                        if sparse_snr > 0.01:
                            status = " ✓"
                        else:
                            status = " (skipped)"
                        print(f"  Sparse: {sparse.num_params:,} params, SNR {sparse_snr:+.2f} dB{status}")

                    # Only add if it improves SNR
                    if sparse_snr > 0.01:
                        stages.append(sparse)
                        current_output = current_output + sparse_output
                        current_weight_approx = current_weight_approx + sparse_weight_full

                except Exception as e:
                    if verbose:
                        print(f"  Sparse: failed ({str(e)[:60]})")

            # Step 4: Kronecker to capture repeating/block-periodic patterns
            if use_kronecker:
                residual_output_4 = original_output - current_output
                residual_weight_4 = W - current_weight_approx

                # Create temporary linear with residual weights
                temp_linear_kron = nn.Linear(in_features, out_features, bias=False)
                temp_linear_kron.weight.data = residual_weight_4.to(dtype)

                try:
                    # Create KroneckerLinear (verbose=False)
                    kronecker = KroneckerLinear.from_linear(
                        temp_linear_kron,
                        factor_size=kronecker_factor_size,
                        verbose=False,
                    )

                    # Move to correct device
                    kronecker = kronecker.to(device)

                    with torch.no_grad():
                        kronecker_output = kronecker(input_activations.to(dtype)).float()

                    kronecker_weight = kronecker.to_matrix().float().to(device)
                    kronecker_snr = compute_snr(residual_output_4, kronecker_output)

                    if verbose:
                        status = ""
                        if kronecker_snr > 0.01:
                            status = " ✓"
                        else:
                            status = " (skipped)"
                        print(f"  Kronecker: {kronecker.num_params:,} params, SNR {kronecker_snr:+.2f} dB{status}")

                    # Only add if it improves SNR
                    if kronecker_snr > 0.01:
                        stages.append(kronecker)
                        current_output = current_output + kronecker_output
                        current_weight_approx = current_weight_approx + kronecker_weight

                except Exception as e:
                    if verbose:
                        print(f"  Kronecker: failed ({str(e)[:60]})")

        # Final SNR
        final_snr = compute_snr(original_output, current_output)
        total_params = sum(
            stage.num_params for stage in stages
        )
        original_params = W.numel()

        if verbose:
            print(f"\nFinal Results:")
            print(f"  Stages: {len(stages)}")
            print(f"  Final SNR: {final_snr:.2f} dB")
            print(f"  Total params: {total_params:,} ({total_params/original_params:.2f}x)")
            print(f"  Compression: {100*(1-total_params/original_params):.1f}%")

        # Create module and ensure it's on the correct device
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
        # Determine shape from first stage
        if not self.stages:
            return None

        # Get dimensions
        first_stage = self.stages[0]
        if isinstance(first_stage, LowRankLinear):
            out_features = first_stage.out_features
            in_features = first_stage.in_features
        elif hasattr(first_stage, 'out_features'):
            # TuckerLinear, ColumnSparseLinear, etc.
            out_features = first_stage.out_features
            in_features = first_stage.in_features
        else:
            return None

        # Reconstruct
        device = first_stage.weight.device if hasattr(first_stage, 'weight') else 'cpu'
        total = torch.zeros(out_features, in_features, device=device)

        for stage in self.stages:
            if hasattr(stage, 'to_matrix'):
                total = total + stage.to_matrix()
            elif isinstance(stage, LowRankLinear):
                total = total + (stage.U.data @ stage.V.data.T)
            elif hasattr(stage, 'selected_columns'):
                # ColumnSparseLinear - reconstruct sparse matrix
                sparse_weight = torch.zeros(out_features, in_features, device=device)
                sparse_weight[:, stage.selected_columns] = stage.weight.data
                total = total + sparse_weight

        return total


__all__ = ['GreedyMultiScaleLinear', 'LowRankLinear']
