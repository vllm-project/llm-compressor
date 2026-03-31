"""Kronecker Product Decomposition for Linear Layers.

Decomposes a weight matrix via Kronecker product:
    W ≈ B ⊗ C

where B and C are smaller matrices, and ⊗ is the Kronecker product.

Mathematical form:
    For W of shape (m*n, p*q):
        W ≈ B ⊗ C  where B is (m, p) and C is (n, q)
        (B ⊗ C)[i*n + j, k*q + l] = B[i, k] * C[j, l]

The Geometry:
    - Captures repeating or block-periodic structure
    - "Fractal-like" - small pattern repeated to create large matrix
    - Common in convolutional layers and attention patterns

Benefits:
    - Incredibly parameter-efficient: O(mp + nq) instead of O(mnpq)
    - For 1024x1024 matrix: two 32x32 matrices = 2,048 params (vs 1M)
    - Leaves massive budget for residual corrections
    - Perfect for matrices with repeating structure

Example:
    1024x1024 matrix → 32x32 ⊗ 32x32 = 2,048 parameters (0.2%)
"""

import torch
import torch.nn as nn
from typing import Optional
import math


class KroneckerLinear(nn.Module):
    """Linear layer compressed via Kronecker product decomposition.

    W ≈ B ⊗ C where B and C are smaller factor matrices.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        B: torch.Tensor,
        C: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            B: First Kronecker factor (m, p)
            C: Second Kronecker factor (n, q)
            bias: Optional bias
            dtype: Data type
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # B: (m, p), C: (n, q)
        # W = B ⊗ C has shape (m*n, p*q)
        self.m, self.p = B.shape
        self.n, self.q = C.shape

        # Register factors as parameters
        self.B = nn.Parameter(B.to(dtype))
        self.C = nn.Parameter(C.to(dtype))

        if bias is not None:
            self.bias = nn.Parameter(bias.to(dtype))
        else:
            self.register_parameter('bias', None)

        # Learnable scaling
        self.alpha = nn.Parameter(torch.ones(1, dtype=dtype))

    @property
    def num_params(self):
        """Count parameters."""
        total = self.B.numel() + self.C.numel()
        total += self.alpha.numel()
        if self.bias is not None:
            total += self.bias.numel()
        return total

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        factor_size: Optional[int] = None,
        verbose: bool = False,
    ):
        """Create KroneckerLinear from standard Linear layer.

        Args:
            linear: Original linear layer
            factor_size: Size of Kronecker factors (default: sqrt of dims)
            verbose: Print decomposition info
        """
        weight = linear.weight.detach().clone()
        bias = linear.bias.detach().clone() if linear.bias is not None else None

        out_features, in_features = weight.shape
        orig_dtype = weight.dtype
        device = weight.device

        # Determine factor sizes
        # For W: (out_features, in_features) = (m*n, p*q)
        # We want B: (m, p) and C: (n, q)
        # Choose m, n, p, q such that m*n = out_features, p*q = in_features
        # Default: make them roughly square

        if factor_size is None:
            # Use sqrt as default
            m = n = int(math.sqrt(out_features))
            p = q = int(math.sqrt(in_features))

            # Adjust to make divisible
            while out_features % m != 0 and m > 1:
                m -= 1
            n = out_features // m

            while in_features % p != 0 and p > 1:
                p -= 1
            q = in_features // p
        else:
            m = n = factor_size
            p = q = factor_size
            # Adjust if doesn't divide evenly
            while out_features % m != 0 and m > 1:
                m -= 1
            n = out_features // m
            while in_features % p != 0 and p > 1:
                p -= 1
            q = in_features // p

        if verbose:
            orig_params = weight.numel()
            kron_params = m * p + n * q
            print(f"Kronecker decomposition:")
            print(f"  Original: {weight.shape}")
            print(f"  Factors: B({m}, {p}) ⊗ C({n}, {q})")
            print(f"  Params: {kron_params:,} / {orig_params:,} = {kron_params/orig_params:.2%}")

        # Fit Kronecker decomposition
        # W ≈ B ⊗ C
        # Use least squares to find B and C

        weight_float = weight.float().cpu()

        # Reshape W to (m, n, p, q) then to (m*p, n*q)
        # This allows us to factor out B and C
        try:
            W_reshaped = weight_float.reshape(m, n, p, q)
            W_perm = W_reshaped.permute(0, 2, 1, 3).reshape(m * p, n * q)

            # Initialize B and C with SVD-based approximation
            # Approximate W_perm ≈ vec(B) @ vec(C)^T
            # Use rank-1 SVD
            U, S, Vh = torch.linalg.svd(W_perm, full_matrices=False)

            # Take rank-1 approximation
            u1 = U[:, 0] * math.sqrt(S[0])
            v1 = Vh[0, :] * math.sqrt(S[0])

            # Reshape to factors
            B = u1.reshape(m, p)
            C = v1.reshape(n, q)

        except:
            # Fallback: random initialization
            if verbose:
                print("  Using random initialization (reshape failed)")
            B = torch.randn(m, p) * 0.01
            C = torch.randn(n, q) * 0.01

        if verbose:
            # Measure reconstruction quality
            W_approx = torch.kron(B, C)
            # Truncate or pad to match original size
            W_approx_sized = W_approx[:out_features, :in_features]
            W_orig_sized = weight_float[:W_approx_sized.shape[0], :W_approx_sized.shape[1]]

            mse = torch.mean((W_orig_sized - W_approx_sized) ** 2).item()
            signal_var = torch.var(weight_float).item()
            rel_error = (mse / signal_var) ** 0.5
            print(f"  Reconstruction relative error: {rel_error:.4f}")

        return cls(
            in_features=in_features,
            out_features=out_features,
            B=B,
            C=C,
            bias=bias,
            dtype=orig_dtype,
        ).to(device)

    def forward(self, x):
        """Forward pass using Kronecker product structure.

        Efficient computation: (B ⊗ C) @ x = vec(C @ X @ B^T)
        where X is x reshaped appropriately.
        """
        # Store original shape
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        # Flatten batch dimensions
        x_flat = x.reshape(-1, self.in_features)
        batch_size = x_flat.shape[0]

        # Kronecker product forward: y = (B ⊗ C) @ x
        # Efficient formula: vec((B ⊗ C) @ x) = vec(C @ X @ B^T)
        # where X = reshape(x, (batch, n, q))

        # Reshape input: (batch, p*q) -> (batch, q, p)
        if self.in_features == self.p * self.q:
            X = x_flat.reshape(batch_size, self.q, self.p)
        else:
            # Pad if needed
            pad_size = self.p * self.q - self.in_features
            if pad_size > 0:
                x_padded = torch.cat([
                    x_flat,
                    torch.zeros(batch_size, pad_size, dtype=x.dtype, device=x.device)
                ], dim=1)
                X = x_padded.reshape(batch_size, self.q, self.p)
            else:
                X = x_flat[:, :self.p * self.q].reshape(batch_size, self.q, self.p)

        # Compute: Y = C @ X @ B^T
        # X: (batch, q, p)
        # B^T: (p, m)
        # X @ B^T: (batch, q, m)
        # C: (n, q)
        # C @ (X @ B^T).transpose: (batch, n, m)

        XB = X @ self.B.to(x.dtype).T  # (batch, q, m)
        Y = XB.permute(0, 2, 1) @ self.C.to(x.dtype).T  # (batch, m, n)

        # Reshape output: (batch, m, n) -> (batch, m*n)
        result = Y.reshape(batch_size, self.m * self.n)

        # Truncate to output size
        result = result[:, :self.out_features]

        # Apply scaling
        result = result * self.alpha.to(x.dtype)

        # Add bias
        if self.bias is not None:
            result = result + self.bias.to(x.dtype)

        # Reshape back to original batch dimensions
        result = result.reshape(*batch_dims, self.out_features)

        return result

    def to_matrix(self) -> torch.Tensor:
        """Reconstruct full weight matrix from Kronecker factors."""
        # W = B ⊗ C
        W = torch.kron(self.B, self.C)

        # Truncate to actual size
        W = W[:self.out_features, :self.in_features]

        # Apply scaling
        W = W * self.alpha

        return W


__all__ = ['KroneckerLinear']
