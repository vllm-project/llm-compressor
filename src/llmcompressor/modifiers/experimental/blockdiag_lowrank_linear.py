"""Block-Diagonal + Low-Rank Decomposition (Monarch-style).

Decomposes a weight matrix into:
    W = BlockDiag + LowRank

where:
    - BlockDiag: Block-diagonal matrix with dense blocks (local structure)
    - LowRank: Global low-rank correction (inter-block communication)

Mathematical form:
    y = BlockDiag(x) + U @ V^T @ x

The Geometry:
    - Block-diagonal captures LOCAL structure (neuron clusters)
    - Each block is dense → perfect reconstruction of local features
    - Low-rank handles GLOBAL communication between blocks
    - "Divide and conquer" approach

Benefits:
    - More stable SNR than pure low-rank (preserves local detail)
    - Efficient: O(B * (N/B)^2) + O(r*N) where B = num blocks, r = rank
    - Natural decomposition: local features + global coordination
    - Inspired by Monarch decompositions
"""

import torch
import torch.nn as nn
from typing import Optional
import math


class BlockDiagonalLowRankLinear(nn.Module):
    """Linear layer as Block-Diagonal + Low-Rank.

    Combines local structure (dense blocks) with global communication (low-rank).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_blocks: int,
        rank: int,
        blocks: list[torch.Tensor],
        U: torch.Tensor,
        V: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            num_blocks: Number of diagonal blocks
            rank: Rank of low-rank component
            blocks: List of dense block matrices
            U: Low-rank output factor (out_features, rank)
            V: Low-rank input factor (in_features, rank)
            bias: Optional bias
            dtype: Data type
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        self.rank = rank

        # Block sizes
        self.block_in_size = in_features // num_blocks
        self.block_out_size = out_features // num_blocks

        # Register blocks as parameters
        self.blocks = nn.ParameterList([nn.Parameter(b.to(dtype)) for b in blocks])

        # Low-rank factors
        self.U = nn.Parameter(U.to(dtype))
        self.V = nn.Parameter(V.to(dtype))

        if bias is not None:
            self.bias = nn.Parameter(bias.to(dtype))
        else:
            self.register_parameter('bias', None)

        # Learnable scaling
        self.alpha = nn.Parameter(torch.ones(1, dtype=dtype))

    @property
    def num_params(self):
        """Count parameters."""
        total = sum(b.numel() for b in self.blocks)
        total += self.U.numel() + self.V.numel()
        total += self.alpha.numel()
        if self.bias is not None:
            total += self.bias.numel()
        return total

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        num_blocks: int = 16,
        rank: int = 64,
        verbose: bool = False,
    ):
        """Create BlockDiagonalLowRankLinear from standard Linear layer.

        Args:
            linear: Original linear layer
            num_blocks: Number of diagonal blocks
            rank: Rank of low-rank component
            verbose: Print decomposition info
        """
        weight = linear.weight.detach().clone()
        bias = linear.bias.detach().clone() if linear.bias is not None else None

        out_features, in_features = weight.shape
        orig_dtype = weight.dtype
        device = weight.device

        # Check that dimensions are divisible by num_blocks
        if in_features % num_blocks != 0 or out_features % num_blocks != 0:
            # Adjust num_blocks to nearest divisor
            for nb in range(num_blocks, 0, -1):
                if in_features % nb == 0 and out_features % nb == 0:
                    if verbose:
                        print(f"  Adjusting num_blocks from {num_blocks} to {nb} for divisibility")
                    num_blocks = nb
                    break

        block_in_size = in_features // num_blocks
        block_out_size = out_features // num_blocks

        if verbose:
            orig_params = weight.numel()
            block_params = num_blocks * block_in_size * block_out_size
            lr_params = rank * (in_features + out_features)
            total_params = block_params + lr_params

            print(f"Block-Diagonal + Low-Rank decomposition:")
            print(f"  Original: {weight.shape}")
            print(f"  Num blocks: {num_blocks}")
            print(f"  Block size: ({block_out_size}, {block_in_size})")
            print(f"  Low-rank: {rank}")
            print(f"  Block params: {block_params:,}")
            print(f"  LR params: {lr_params:,}")
            print(f"  Total params: {total_params:,} / {orig_params:,} = {total_params/orig_params:.2%}")

        # Extract diagonal blocks from weight matrix
        weight_float = weight.float().cpu()
        blocks = []

        for i in range(num_blocks):
            # Extract block from diagonal
            row_start = i * block_out_size
            row_end = (i + 1) * block_out_size
            col_start = i * block_in_size
            col_end = (i + 1) * block_in_size

            block = weight_float[row_start:row_end, col_start:col_end].clone()
            blocks.append(block)

        # Compute residual after removing block-diagonal
        # Reconstruct block-diagonal matrix
        block_diag = torch.zeros_like(weight_float)
        for i, block in enumerate(blocks):
            row_start = i * block_out_size
            row_end = (i + 1) * block_out_size
            col_start = i * block_in_size
            col_end = (i + 1) * block_in_size
            block_diag[row_start:row_end, col_start:col_end] = block

        # Residual for low-rank approximation
        residual = weight_float - block_diag

        # Fit low-rank to residual via SVD
        U_full, S, Vh = torch.linalg.svd(residual, full_matrices=False)
        U_r = U_full[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]

        U = U_r @ torch.diag(S_r)
        V = Vh_r.T

        if verbose:
            # Measure reconstruction quality
            W_approx = block_diag + (U @ V.T)
            mse = torch.mean((weight_float - W_approx) ** 2).item()
            signal_var = torch.var(weight_float).item()
            rel_error = (mse / signal_var) ** 0.5
            print(f"  Reconstruction relative error: {rel_error:.4f}")

        return cls(
            in_features=in_features,
            out_features=out_features,
            num_blocks=num_blocks,
            rank=rank,
            blocks=blocks,
            U=U,
            V=V,
            bias=bias,
            dtype=orig_dtype,
        ).to(device)

    def forward(self, x):
        """Forward pass: y = BlockDiag(x) + U @ V^T @ x"""
        # Store original shape
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        # Flatten batch dimensions
        x_flat = x.reshape(-1, self.in_features)

        # Block-diagonal computation
        # Split input into blocks
        x_blocks = x_flat.reshape(-1, self.num_blocks, self.block_in_size)

        # Apply each block
        y_blocks = []
        for i, block in enumerate(self.blocks):
            # x_blocks[:, i, :] has shape (batch, block_in_size)
            # block has shape (block_out_size, block_in_size)
            y_block = x_blocks[:, i, :] @ block.to(x.dtype).T
            y_blocks.append(y_block)

        # Concatenate block outputs
        y_blockdiag = torch.cat(y_blocks, dim=-1)

        # Low-rank computation: x @ V @ U^T
        V_dtype = self.V.to(x.dtype)
        U_dtype = self.U.to(x.dtype)
        y_lowrank = x_flat @ V_dtype @ U_dtype.T

        # Combine
        result = y_blockdiag + y_lowrank

        # Apply scaling
        result = result * self.alpha.to(x.dtype)

        # Add bias
        if self.bias is not None:
            result = result + self.bias.to(x.dtype)

        # Reshape back to original batch dimensions
        result = result.reshape(*batch_dims, self.out_features)

        return result

    def to_matrix(self) -> torch.Tensor:
        """Reconstruct full weight matrix."""
        # Reconstruct block-diagonal
        block_diag = torch.zeros(
            self.out_features,
            self.in_features,
            dtype=self.blocks[0].dtype,
            device=self.blocks[0].device
        )

        for i, block in enumerate(self.blocks):
            row_start = i * self.block_out_size
            row_end = (i + 1) * self.block_out_size
            col_start = i * self.block_in_size
            col_end = (i + 1) * self.block_in_size
            block_diag[row_start:row_end, col_start:col_end] = block

        # Add low-rank component
        W = block_diag + (self.U @ self.V.T)

        # Apply scaling
        W = W * self.alpha

        return W


__all__ = ['BlockDiagonalLowRankLinear']
