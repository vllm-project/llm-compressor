"""Tucker Decomposition for Linear Layers.

Tucker decomposition factorizes a tensor into:
    T = G ×₁ U₁ ×₂ U₂ ... ×ₙ Uₙ

where:
    - G is a small core tensor
    - Uᵢ are factor matrices for each mode

For weight matrix W (out_features, in_features):
    1. Reshape to higher-order tensor (n₀, n₁, ..., m₀, m₁, ...)
    2. Decompose: W_tensor = G ×₁ U_out₀ ×₂ U_out₁ ... ×ₖ U_in₀ ×ₖ₊₁ U_in₁ ...
    3. Forward: contract input with factor matrices and core

Benefits over Tensor Train:
    - More compact for same accuracy (dense core vs chain of tensors)
    - Fewer parameters for similar reconstruction quality
    - Faster inference (fewer sequential operations)
"""

import torch
import torch.nn as nn
import tensorly as tl
from typing import Optional
import math


def get_nearest_power_of_2(x: float) -> int:
    """Round to nearest power of 2."""
    return 2 ** round(math.log2(max(1, x)))


class TuckerLinear(nn.Module):
    """Linear layer compressed via Tucker decomposition.

    The weight matrix is reshaped to a higher-order tensor and decomposed:
        W_tensor = core ×₁ U_out[0] ×₂ U_out[1] ... ×ₖ U_in[0] ×ₖ₊₁ U_in[1] ...
    """

    def __init__(
        self,
        core: torch.Tensor,
        factors_out: list[torch.Tensor],
        factors_in: list[torch.Tensor],
        output_shape: tuple[int, ...],
        input_shape: tuple[int, ...],
        bias: Optional[torch.Tensor] = None,
        input_perm: Optional[torch.Tensor] = None,
        input_inv_perm: Optional[torch.Tensor] = None,
        output_perm: Optional[torch.Tensor] = None,
        output_inv_perm: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            core: Core tensor of shape (*core_shape_out, *core_shape_in)
            factors_out: List of factor matrices for output modes
            factors_in: List of factor matrices for input modes
            output_shape: Shape for output dimension factorization
            input_shape: Shape for input dimension factorization
            bias: Optional bias term
            input_perm: Optional input channel permutation (for spectral reordering)
            input_inv_perm: Inverse of input_perm
            output_perm: Optional output channel permutation (for spectral reordering)
            output_inv_perm: Inverse of output_perm
            dtype: Data type for parameters
        """
        super().__init__()

        self.output_shape = output_shape
        self.input_shape = input_shape
        self.out_features = math.prod(output_shape)
        self.in_features = math.prod(input_shape)

        # Register core and factors as parameters
        self.core = nn.Parameter(core.to(dtype))
        self.factors_out = nn.ParameterList([nn.Parameter(f.to(dtype)) for f in factors_out])
        self.factors_in = nn.ParameterList([nn.Parameter(f.to(dtype)) for f in factors_in])

        if bias is not None:
            self.bias = nn.Parameter(bias.to(dtype))
        else:
            self.register_parameter('bias', None)

        # Learnable scaling parameters (like TensorizedLinear)
        self.alpha = nn.Parameter(torch.ones(1, dtype=dtype))
        self.per_dim_scale = nn.Parameter(torch.ones(self.out_features, dtype=dtype))

        # Permutation buffers
        if input_perm is not None:
            self.register_buffer('input_perm', input_perm)
            self.register_buffer('input_inv_perm', input_inv_perm)
        else:
            self.register_buffer('input_perm', None)
            self.register_buffer('input_inv_perm', None)

        if output_perm is not None:
            self.register_buffer('output_perm', output_perm)
            self.register_buffer('output_inv_perm', output_inv_perm)
        else:
            self.register_buffer('output_perm', None)
            self.register_buffer('output_inv_perm', None)

    @property
    def num_params(self):
        """Count parameters (excluding permutations)."""
        total = self.core.numel()
        total += sum(f.numel() for f in self.factors_out)
        total += sum(f.numel() for f in self.factors_in)
        if self.bias is not None:
            total += self.bias.numel()
        total += self.alpha.numel()
        total += self.per_dim_scale.numel()
        return total

    @staticmethod
    def _spectral_reordering_input(
        weight: torch.Tensor,
        input_activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Use spectral reordering to find optimal input channel permutation.

        Groups correlated input channels together using Laplacian eigenmaps.
        """
        if input_activations is not None:
            # Use activation-based correlation
            activations_centered = input_activations - input_activations.mean(dim=0, keepdim=True)
            activations_std = activations_centered.std(dim=0, keepdim=True) + 1e-10
            activations_normalized = activations_centered / activations_std
            num_samples = activations_normalized.shape[0]
            correlation = (activations_normalized.T @ activations_normalized) / num_samples
        else:
            # Use weight-based correlation (input features)
            weight_normalized = weight / (torch.norm(weight, dim=0, keepdim=True) + 1e-10)
            correlation = weight_normalized.T @ weight_normalized

        # Exponential kernel for stronger locality
        affinity = torch.exp(correlation - 1.0)

        # Graph Laplacian
        degree = torch.diag(affinity.sum(dim=1))
        laplacian = degree - affinity

        # Eigendecomposition
        laplacian_f32 = laplacian.to(torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_f32)

        # Fiedler vector (2nd smallest eigenvalue)
        fiedler_vector = eigenvectors[:, 1]

        # Sort by Fiedler vector
        input_perm = torch.argsort(fiedler_vector)

        return input_perm.to(weight.device)

    @staticmethod
    def _spectral_reordering_output(
        weight: torch.Tensor,
        output_activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Use spectral reordering to find optimal output channel permutation.

        Groups correlated output channels together using Laplacian eigenmaps.
        """
        if output_activations is not None:
            # Use output activation-based correlation
            activations_centered = output_activations - output_activations.mean(dim=0, keepdim=True)
            activations_std = activations_centered.std(dim=0, keepdim=True) + 1e-10
            activations_normalized = activations_centered / activations_std
            num_samples = activations_normalized.shape[0]
            correlation = (activations_normalized.T @ activations_normalized) / num_samples
        else:
            # Use weight-based correlation (output features = rows of W)
            weight_normalized = weight / (torch.norm(weight, dim=1, keepdim=True) + 1e-10)
            correlation = weight_normalized @ weight_normalized.T

        # Exponential kernel for stronger locality
        affinity = torch.exp(correlation - 1.0)

        # Graph Laplacian
        degree = torch.diag(affinity.sum(dim=1))
        laplacian = degree - affinity

        # Eigendecomposition
        laplacian_f32 = laplacian.to(torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_f32)

        # Fiedler vector (2nd smallest eigenvalue)
        fiedler_vector = eigenvectors[:, 1]

        # Sort by Fiedler vector
        output_perm = torch.argsort(fiedler_vector)

        return output_perm.to(weight.device)

    @staticmethod
    def get_shape(num_features: int, num_modes: int) -> tuple[int, ...]:
        """Compute balanced factorization of num_features into num_modes dimensions.

        Similar to TensorizedLinear.get_shape, tries to use powers of 2.
        """
        shape = []
        remainder = num_features
        for i in reversed(range(num_modes)):
            if i == 0:
                shape.append(round(remainder))
            else:
                dim = get_nearest_power_of_2(remainder ** (1 / (num_modes - len(shape))))
                shape.append(dim)
                remainder = remainder / dim

        assert len(shape) == num_modes
        assert math.prod(shape) == num_features, f"Shape mismatch: {shape} -> {math.prod(shape)} != {num_features}"
        return tuple(shape)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: float | tuple[int, ...] = 0.5,
        num_modes: int = 3,
        input_activations: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ):
        """Create TuckerLinear from standard Linear layer.

        Args:
            linear: Original linear layer
            rank: Tucker rank specification
                - float: rank ratio (e.g., 0.5 means core dims are ~0.5 of original)
                - tuple: explicit core dimensions for each mode
            num_modes: Number of modes for each dimension (output and input)
            input_activations: Optional activations for spectral reordering
            verbose: Print decomposition info
        """
        weight = linear.weight.detach().clone()
        bias = linear.bias.detach().clone() if linear.bias is not None else None

        out_features, in_features = weight.shape
        orig_dtype = weight.dtype
        device = weight.device

        # Spectral reordering for BOTH input and output dimensions
        # This is critical for Tucker - we factorize both dimensions
        input_perm = cls._spectral_reordering_input(weight, input_activations)
        input_inv_perm = torch.argsort(input_perm)

        # For output, we don't have output activations, so use weight-based correlation
        output_perm = cls._spectral_reordering_output(weight, None)
        output_inv_perm = torch.argsort(output_perm)

        # Apply both permutations
        weight_permuted = weight[output_perm, :][:, input_perm]

        if verbose:
            print(f"Spectral reordering:")
            print(f"  Input permutation computed (groups correlated input features)")
            print(f"  Output permutation computed (groups correlated output features)")

        # Get factorization shapes
        output_shape = cls.get_shape(out_features, num_modes)
        input_shape = cls.get_shape(in_features, num_modes)
        tensor_shape = output_shape + input_shape

        # Reshape to tensor
        weight_tensor = weight_permuted.reshape(tensor_shape)

        # Determine Tucker ranks
        if isinstance(rank, float):
            # Compute core dimensions as rank ratio of original dims
            core_dims = tuple(max(1, int(d * rank)) for d in tensor_shape)
        else:
            core_dims = rank

        if verbose:
            print(f"Tucker decomposition:")
            print(f"  Original: {weight.shape}")
            print(f"  Tensor shape: {tensor_shape}")
            print(f"  Core dims: {core_dims}")
            orig_params = weight.numel()
            core_params = math.prod(core_dims)
            factor_params = sum(tensor_shape[i] * core_dims[i] for i in range(len(tensor_shape)))
            total_params = core_params + factor_params
            print(f"  Params: {total_params:,} / {orig_params:,} = {total_params/orig_params:.2%}")

        # Tucker decomposition
        weight_f32 = weight_tensor.to(torch.float32)
        core, factors = tl.decomposition.tucker(weight_f32, rank=core_dims)

        # Split factors into output and input
        factors_out = factors[:num_modes]
        factors_in = factors[num_modes:]

        return cls(
            core=core,
            factors_out=factors_out,
            factors_in=factors_in,
            output_shape=output_shape,
            input_shape=input_shape,
            bias=bias,
            input_perm=input_perm,
            input_inv_perm=input_inv_perm,
            output_perm=output_perm,
            output_inv_perm=output_inv_perm,
            dtype=orig_dtype,
        ).to(device)

    def forward(self, x):
        """Forward pass using Tucker contraction via tensorly.

        Uses tensorly's tucker_to_tensor to reconstruct weight matrix,
        similar to how TensorizedLinear uses tt_matrix_to_tensor.
        """
        # Store original shape
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        # Apply input permutation
        if self.input_perm is not None:
            x = x[..., self.input_perm]

        # Flatten batch dimensions
        x_flat = x.reshape(-1, self.in_features)
        batch_total = x_flat.shape[0]

        # Reconstruct weight matrix from Tucker decomposition
        # Combine output and input factors
        factors = list(self.factors_out) + list(self.factors_in)

        # Reconstruct full tensor
        W_tensor = tl.tucker_to_tensor((self.core.to(x.dtype), [f.to(x.dtype) for f in factors]))

        # Reshape to matrix: (out_features, in_features)
        W = W_tensor.reshape(self.out_features, self.in_features)

        # Apply scaling
        scale = (self.per_dim_scale * self.alpha).to(x.dtype)
        W_scaled = W * scale.view(-1, 1)

        # Apply linear transformation: x @ W^T
        # W is in permuted space, so result will be in permuted output space
        result = x_flat @ W_scaled.T

        # Apply inverse output permutation to restore original output ordering
        if self.output_inv_perm is not None:
            result = result[..., self.output_inv_perm]

        # Add bias
        if self.bias is not None:
            result = result + self.bias.to(x.dtype)

        # Reshape back to original batch dimensions
        result = result.reshape(*batch_dims, self.out_features)

        return result

    def to_matrix(self) -> torch.Tensor:
        """Reconstruct full weight matrix with scaling and inverse permutation."""
        # Reconstruct from Tucker decomposition
        factors = list(self.factors_out) + list(self.factors_in)
        W_tensor = tl.tucker_to_tensor((self.core, factors))

        # Reshape to matrix
        W = W_tensor.reshape(self.out_features, self.in_features)

        # Apply scaling
        scale = self.per_dim_scale * self.alpha
        W_scaled = W * scale.view(-1, 1)

        # Apply inverse permutations to restore original ordering
        # W is in (permuted_output, permuted_input) space
        # Need to unpermute both dimensions
        if self.input_inv_perm is not None:
            W_scaled = W_scaled[:, self.input_inv_perm]

        if self.output_inv_perm is not None:
            W_scaled = W_scaled[self.output_inv_perm, :]

        return W_scaled


__all__ = ['TuckerLinear']
