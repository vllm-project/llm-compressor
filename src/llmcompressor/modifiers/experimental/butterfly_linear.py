"""Butterfly Matrix Decomposition for Linear Layers.

Butterfly matrices decompose a weight matrix into a product of log(N) sparse matrices,
following the structure of the Fast Fourier Transform (FFT).

Mathematical form:
    W = B_1 @ B_2 @ ... @ B_log(N)

where each B_i is a sparse matrix with a specific butterfly pattern.

Key properties:
    - O(N log N) parameters instead of O(N^2)
    - Preserves high-frequency information (unlike low-rank)
    - Hierarchical structure captures recursive relationships
    - Inspired by FFT, Monarch, and Pixelfly architectures

Benefits:
    - Excellent for preserving sharp features and edges
    - Complements low-rank (which captures flat correlations)
    - Low-rank kills high-freq → Butterfly preserves high-freq
    - Together they cover different parts of the frequency spectrum
"""

import torch
import torch.nn as nn
from typing import Optional
import math


class ButterflyLinear(nn.Module):
    """Linear layer compressed via Butterfly matrix factorization.

    Decomposes weight into product of log2(N) sparse matrices following
    the FFT butterfly pattern.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_factors: Optional[int] = None,
        bias: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            num_factors: Number of butterfly factors (default: log2(max(in, out)))
            bias: Optional bias term
            dtype: Data type for parameters
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Determine number of factors (stages in butterfly)
        if num_factors is None:
            # Default: log2 of the larger dimension
            num_factors = math.ceil(math.log2(max(in_features, out_features)))

        self.num_factors = num_factors

        # Pad dimensions to nearest power of 2 for clean butterfly structure
        self.padded_in = 2 ** math.ceil(math.log2(in_features))
        self.padded_out = 2 ** math.ceil(math.log2(out_features))

        # Each factor is a block-diagonal matrix with 2x2 blocks
        # For a dimension N, we have N/2 blocks of size 2x2
        # Total params per factor: N * 2 (each 2x2 block has 4 params, N/2 blocks)

        # Create butterfly factors from input to intermediate to output
        # Structure: in → intermediate stages → out

        # First factor: padded_in → padded_in (permutation + mixing)
        # Middle factors: padded_in → padded_in
        # Last factor: padded_in → padded_out (dimension change)

        self.factors = nn.ParameterList()

        for i in range(num_factors):
            if i == num_factors - 1:
                # Last factor handles dimension change
                # Create a factor from padded_in to padded_out
                factor_shape = (self.padded_out, self.padded_in)
            else:
                # Intermediate factors maintain dimension
                factor_shape = (self.padded_in, self.padded_in)

            # Each factor is stored as a list of 2x2 blocks
            # For dimension N, we have N/2 blocks along the diagonal
            num_blocks = factor_shape[0] // 2

            # Store as (num_blocks, 2, 2) tensor
            factor = nn.Parameter(torch.randn(num_blocks, 2, 2, dtype=dtype) * 0.01)
            self.factors.append(factor)

        if bias is not None:
            self.bias = nn.Parameter(bias.to(dtype))
        else:
            self.register_parameter('bias', None)

        # Learnable scaling
        self.alpha = nn.Parameter(torch.ones(1, dtype=dtype))

    @property
    def num_params(self):
        """Count parameters."""
        total = sum(f.numel() for f in self.factors)
        total += self.alpha.numel()
        if self.bias is not None:
            total += self.bias.numel()
        return total

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        num_factors: Optional[int] = None,
        num_iters: int = 100,
        lr: float = 0.01,
        verbose: bool = False,
    ):
        """Create ButterflyLinear from standard Linear layer.

        Args:
            linear: Original linear layer
            num_factors: Number of butterfly factors (default: log2(max_dim))
            num_iters: Number of optimization iterations to fit factors
            lr: Learning rate for fitting
            verbose: Print decomposition info
        """
        weight = linear.weight.detach().clone()
        bias = linear.bias.detach().clone() if linear.bias is not None else None

        out_features, in_features = weight.shape
        orig_dtype = weight.dtype
        device = weight.device

        # Determine number of factors
        if num_factors is None:
            num_factors = math.ceil(math.log2(max(in_features, out_features)))

        if verbose:
            padded_in = 2 ** math.ceil(math.log2(in_features))
            padded_out = 2 ** math.ceil(math.log2(out_features))
            orig_params = weight.numel()
            # Each factor has (N/2) blocks of 2x2 = 2*N params
            # Most factors are padded_in x padded_in, last is padded_out x padded_in
            butterfly_params = (num_factors - 1) * 2 * padded_in + 2 * padded_out

            print(f"Butterfly decomposition:")
            print(f"  Original: {weight.shape}")
            print(f"  Padded: ({padded_out}, {padded_in})")
            print(f"  Num factors: {num_factors}")
            print(f"  Params: {butterfly_params:,} / {orig_params:,} = {butterfly_params/orig_params:.2%}")

        # Create butterfly layer
        butterfly = cls(
            in_features=in_features,
            out_features=out_features,
            num_factors=num_factors,
            bias=bias,
            dtype=orig_dtype,
        ).to(device)

        # Initialize factors with random values
        for factor in butterfly.factors:
            # Initialize with small random values
            factor.data = torch.randn_like(factor) * 0.01
            # Add identity component for stability
            factor.data[:, 0, 0] += 1.0
            factor.data[:, 1, 1] += 1.0

        # Fit butterfly factors to approximate weight matrix via gradient descent
        if verbose:
            print(f"  Fitting butterfly factors ({num_iters} iterations)...")

        optimizer = torch.optim.Adam(butterfly.parameters(), lr=lr)
        weight_float = weight.float()

        for iter_idx in range(num_iters):
            optimizer.zero_grad()

            # Reconstruct weight and compute loss
            W_approx = butterfly.to_matrix()
            loss = torch.nn.functional.mse_loss(W_approx, weight_float)

            # Backprop and update
            loss.backward()
            optimizer.step()

            if verbose and (iter_idx + 1) % 20 == 0:
                with torch.no_grad():
                    # Compute reconstruction error
                    rel_error = (loss.item() / weight_float.var().item()) ** 0.5
                    print(f"    Iter {iter_idx + 1}/{num_iters}: loss={loss.item():.6f}, rel_error={rel_error:.4f}")

        if verbose:
            with torch.no_grad():
                W_final = butterfly.to_matrix()
                final_mse = torch.nn.functional.mse_loss(W_final, weight_float).item()
                final_rel_error = (final_mse / weight_float.var().item()) ** 0.5
                print(f"  Final relative error: {final_rel_error:.4f}")

        return butterfly

    def _apply_butterfly_factor(self, x: torch.Tensor, factor: nn.Parameter) -> torch.Tensor:
        """Apply one butterfly factor (block-diagonal 2x2 structure).

        Args:
            x: Input of shape (..., N) where N is even
            factor: Butterfly factor of shape (N/2, 2, 2)

        Returns:
            Output of shape (..., N)
        """
        batch_shape = x.shape[:-1]
        N = x.shape[-1]
        num_blocks = N // 2

        # Reshape to expose 2x2 blocks: (..., N/2, 2)
        x_blocks = x.reshape(*batch_shape, num_blocks, 2)

        # Apply each 2x2 block: (num_blocks, 2, 2) @ (..., num_blocks, 2, 1)
        # Factor: (num_blocks, 2, 2)
        # x_blocks: (..., num_blocks, 2)
        # Want: (..., num_blocks, 2)

        # Expand factor for batch dimensions
        # Use einsum: ...nb,nbc->...nc where b,c are the 2x2 block indices
        output_blocks = torch.einsum('...nb,nbc->...nc', x_blocks, factor)

        # Reshape back to (..., N)
        output = output_blocks.reshape(*batch_shape, N)

        return output

    def forward(self, x):
        """Forward pass through butterfly factorization.

        Applies sequence of sparse butterfly factors.
        """
        # Store original shape
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        # Flatten batch dimensions
        x_flat = x.reshape(-1, self.in_features)

        # Pad to power of 2 if needed
        if self.in_features < self.padded_in:
            padding = torch.zeros(
                x_flat.shape[0],
                self.padded_in - self.in_features,
                dtype=x.dtype,
                device=x.device
            )
            x_flat = torch.cat([x_flat, padding], dim=-1)

        # Apply butterfly factors sequentially
        result = x_flat
        for i, factor in enumerate(self.factors):
            result = self._apply_butterfly_factor(result, factor.to(x.dtype))

        # Truncate to output dimension
        result = result[..., :self.out_features]

        # Apply scaling
        result = result * self.alpha.to(x.dtype)

        # Add bias
        if self.bias is not None:
            result = result + self.bias.to(x.dtype)

        # Reshape back to original batch dimensions
        result = result.reshape(*batch_dims, self.out_features)

        return result

    def to_matrix(self) -> torch.Tensor:
        """Reconstruct full weight matrix from butterfly factors.

        This materializes the full matrix by applying the butterfly
        factorization to the identity matrix.
        """
        device = self.factors[0].device
        dtype = self.factors[0].dtype

        # Create identity matrix for padded input dimension
        I = torch.eye(self.padded_in, dtype=dtype, device=device)

        # Apply all butterfly factors
        result = I
        for factor in self.factors:
            result = self._apply_butterfly_factor(result, factor)

        # Truncate to actual dimensions and transpose (since we applied to columns)
        W = result[:self.padded_out, :self.padded_in].T
        W = W[:self.in_features, :self.out_features].T

        # Apply scaling
        W = W * self.alpha

        return W


__all__ = ['ButterflyLinear']
