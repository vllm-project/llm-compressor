"""Greedy Multi-Scale Decomposition: Cascaded MPO + Low-Rank Residual Corrections.

Mathematical form:
    y = MPO_1(x) + LR_1(x) + MPO_2(x) + LR_2(x) + ...

Strategy:
    1. Fit small MPO_1 (captures global structure)
    2. Compute residual R_1 = Y_true - MPO_1(X)
    3. Fit low-rank LR_1 to R_1 (captures flat correlations)
    4. Compute residual R_2 = R_1 - LR_1(X)
    5. Repeat until target SNR achieved

Benefits:
    - Attacks error from different geometric perspectives
    - More memory efficient than single large MPO
    - Numerically stable (small components)
    - Can achieve high SNR with fewer total parameters
"""

import torch
import torch.nn as nn
from typing import Optional
import sys
import importlib.util

# Load block_tensorized_linear dynamically to avoid import issues
spec_tn = importlib.util.spec_from_file_location(
    "tensorized_linear",
    "src/llmcompressor/modifiers/experimental/tensorized_linear.py"
)
tensorized_module = importlib.util.module_from_spec(spec_tn)
sys.modules['llmcompressor.modifiers.experimental.tensorized_linear'] = tensorized_module
spec_tn.loader.exec_module(tensorized_module)

spec_bt = importlib.util.spec_from_file_location(
    "block_tensorized_linear",
    "src/llmcompressor/modifiers/experimental/block_tensorized_linear.py"
)
block_tn_module = importlib.util.module_from_spec(spec_bt)
spec_bt.loader.exec_module(block_tn_module)

BlockTensorizedLinear = block_tn_module.BlockTensorizedLinear


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
        dtype = weight.dtype

        # Perform SVD on CPU for numerical stability, then move back
        weight_cpu = weight.float().cpu()
        U_full, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)

        # Truncate to rank
        U_r = U_full[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]

        # Create layer
        out_features, in_features = weight.shape
        layer = cls(in_features, out_features, rank)

        # Set parameters: U @ S, V^T and move to correct device
        layer.U.data = (U_r @ torch.diag(S_r)).to(dtype=dtype, device=device)
        layer.V.data = Vh_r.T.to(dtype=dtype, device=device)

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
        return x @ self.V @ self.U.T

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
        mpo_block_size: int = 512,
        mpo_num_cores: int = 3,
        mpo_rank: float = 0.3,  # Low rank for small MPOs
        lr_rank: int = 64,      # Low-rank correction rank
        verbose: bool = True,
    ):
        """Greedily build cascade to reach target SNR.

        Args:
            linear: Original linear layer to approximate
            input_activations: Calibration activations (num_samples, in_features)
            target_snr_db: Target SNR in dB
            max_stages: Maximum number of stages (MPO + LR pairs)
            mpo_block_size: Block size for MPO (should be x^num_cores)
            mpo_num_cores: Number of cores for MPO
            mpo_rank: Rank ratio for each MPO (low for small components)
            lr_rank: Rank for low-rank corrections
            verbose: Print progress
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
            print(f"  MPO config: block_size={mpo_block_size}, num_cores={mpo_num_cores}, rank={mpo_rank}")
            print(f"  LR rank: {lr_rank}\n")

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

            # Step 1: Fit MPO to current residual
            residual_weight = W - current_weight_approx
            residual_output = original_output - current_output

            # Create temporary linear with residual weights on correct device
            temp_linear = nn.Linear(in_features, out_features, bias=False).to(device)
            temp_linear.weight.data = residual_weight.to(dtype)

            # Check if block_size divides dimensions
            if out_features % mpo_block_size == 0 and in_features % mpo_block_size == 0:
                try:
                    mpo = BlockTensorizedLinear.from_linear(
                        temp_linear,
                        block_size=mpo_block_size,
                        rank=mpo_rank,
                        num_cores=mpo_num_cores,
                        input_activations=input_activations,
                    )

                    with torch.no_grad():
                        mpo_output = mpo(input_activations.to(dtype)).float()

                    mpo_weight = mpo.to_matrix().float().to(device)
                    mpo_snr = compute_snr(residual_output, mpo_output)

                    stages.append(mpo)
                    current_output = current_output + mpo_output
                    current_weight_approx = current_weight_approx + mpo_weight

                    if verbose:
                        print(f"  MPO: {mpo.num_params:,} params, SNR improvement: {mpo_snr:.2f} dB")

                except Exception as e:
                    if verbose:
                        print(f"  MPO failed: {e}, skipping to LR")
            else:
                if verbose:
                    print(f"  MPO skipped (block_size doesn't divide dimensions)")

            # Step 2: Fit low-rank to remaining residual
            residual_output_2 = original_output - current_output
            residual_weight_2 = W - current_weight_approx

            # Fit low-rank via SVD of weight residual (already handles device in from_svd)
            lr = LowRankLinear.from_svd(residual_weight_2, rank=lr_rank)

            with torch.no_grad():
                lr_output = lr(input_activations).float()

            lr_weight = (lr.U.data @ lr.V.data.T).float()
            lr_snr = compute_snr(residual_output_2, lr_output)

            stages.append(lr)
            current_output = current_output + lr_output
            current_weight_approx = current_weight_approx + lr_weight

            if verbose:
                print(f"  LR:  {lr.num_params:,} params, SNR improvement: {lr_snr:.2f} dB")

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
        total = 0
        for stage in self.stages:
            if hasattr(stage, 'to_matrix'):
                total = total + stage.to_matrix()
            elif isinstance(stage, LowRankLinear):
                total = total + (stage.U @ stage.V.T)
        return total


__all__ = ['GreedyMultiScaleLinear', 'LowRankLinear']
