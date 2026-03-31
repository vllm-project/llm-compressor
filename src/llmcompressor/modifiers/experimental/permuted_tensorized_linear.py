"""Permuted TensorizedLinear with activation-based spectral reordering.

Uses input activations to determine feature correlations, then reorders
the weight matrix so highly entangled features are adjacent in the MPO.

This dramatically improves MPO efficiency - correlated features can be
captured with lower bond dimension when they're next to each other.
"""

import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from scipy.sparse.csgraph import reverse_cuthill_mckee

from .tensorized_linear import TensorizedLinear


def compute_feature_correlations(activations: torch.Tensor) -> torch.Tensor:
    """Compute correlation matrix between input features.

    Args:
        activations: (num_samples, num_features)

    Returns:
        correlation matrix (num_features, num_features)
    """
    # Center activations
    activations = activations - activations.mean(dim=0, keepdim=True)

    # Compute correlation matrix
    # corr = A^T @ A / n
    corr = (activations.T @ activations) / activations.shape[0]

    return corr.abs()  # Use absolute correlations


def spectral_reordering(corr_matrix: torch.Tensor) -> torch.Tensor:
    """Find permutation that groups correlated features together.

    Uses Reverse Cuthill-McKee algorithm to minimize bandwidth of
    correlation matrix - puts strongly correlated features adjacent.

    Args:
        corr_matrix: (num_features, num_features) correlation matrix

    Returns:
        permutation: (num_features,) indices for reordering
    """
    # Convert to sparse adjacency matrix
    # Threshold to keep only strong correlations
    threshold = corr_matrix.mean() + 0.5 * corr_matrix.std()
    adj = (corr_matrix > threshold).cpu().numpy().astype(float)

    # Reverse Cuthill-McKee reordering
    # This minimizes matrix bandwidth, grouping connected nodes
    try:
        perm = reverse_cuthill_mckee(adj)
        perm = torch.from_numpy(perm).long()
    except:
        # Fallback: use simple correlation-based ordering
        # Sum of correlations per feature
        corr_sum = corr_matrix.sum(dim=1)
        perm = torch.argsort(corr_sum, descending=True)

    return perm


class PermutedTensorizedLinear(nn.Module):
    """TensorizedLinear with learned input permutation for better entanglement.

    The permutation groups highly correlated input features together,
    allowing the MPO to capture entanglement more efficiently.
    """

    def __init__(
        self,
        tensorized_linear: TensorizedLinear,
        input_permutation: torch.Tensor,
        output_permutation: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.tensorized_linear = tensorized_linear

        # Store permutations as buffers (non-trainable but part of state_dict)
        self.register_buffer('input_perm', input_permutation)
        if output_permutation is not None:
            self.register_buffer('output_perm', output_permutation)
        else:
            self.register_buffer('output_perm', None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        input_activations: Optional[torch.Tensor] = None,
        num_cores: int = 3,
        rank: float = 0.5,
        permute_inputs: bool = True,
        permute_outputs: bool = False,
        verbose: bool = False,
    ):
        """Create permuted tensorized linear from regular linear layer.

        Args:
            linear: Original linear layer
            input_activations: Activations for computing correlations (num_samples, in_features)
            num_cores: Number of cores for tensor train
            rank: Rank ratio for TT decomposition
            permute_inputs: Whether to permute input features
            permute_outputs: Whether to permute output features
            verbose: Print reordering info
        """
        device = linear.weight.device
        dtype = linear.weight.dtype
        in_features = linear.in_features
        out_features = linear.out_features

        W = linear.weight.data.clone()

        # Compute input permutation from activations
        input_perm = None
        if permute_inputs and input_activations is not None:
            if verbose:
                print(f"  Computing input feature correlations...")

            acts = input_activations.float()
            corr = compute_feature_correlations(acts)
            input_perm = spectral_reordering(corr)

            if verbose:
                # Measure bandwidth reduction
                orig_bandwidth = (corr != 0).float().sum()
                perm_corr = corr[input_perm][:, input_perm]

                # Compute bandwidth: max distance from diagonal
                n = corr.shape[0]
                indices = torch.arange(n)
                i, j = torch.meshgrid(indices, indices, indexing='ij')
                distances = torch.abs(i - j)

                orig_bw = (distances * (corr != 0).float()).max()
                perm_bw = (distances * (perm_corr != 0).float()).max()

                print(f"  Input reordering: bandwidth {orig_bw:.0f} -> {perm_bw:.0f}")

            # Apply input permutation to weight matrix
            W = W[:, input_perm]
        else:
            input_perm = torch.arange(in_features, device=device)

        # Compute output permutation (optional, less important)
        output_perm = None
        if permute_outputs:
            # Use weight magnitude as proxy for output correlations
            output_corr = (W @ W.T).abs()
            output_perm = spectral_reordering(output_corr)

            if verbose:
                print(f"  Output reordering computed")

            # Apply output permutation
            W = W[output_perm, :]
        else:
            output_perm = torch.arange(out_features, device=device)

        # Create temporary linear with permuted weights
        temp_linear = nn.Linear(in_features, out_features, bias=False)
        temp_linear.weight.data = W.to(dtype)

        # Create tensorized version (on CPU, then move to device)
        temp_linear = temp_linear.cpu()
        tensorized = TensorizedLinear.from_linear(
            temp_linear,
            num_cores=num_cores,
            rank=rank,
        )
        tensorized = tensorized.to(device)

        return cls(tensorized, input_perm.to(device), output_perm.to(device) if permute_outputs else None)

    def forward(self, x):
        """Forward with permutation."""
        # Apply input permutation
        x_perm = x[..., self.input_perm]

        # Apply tensorized linear
        out = self.tensorized_linear(x_perm)

        # Reverse output permutation if present
        if self.output_perm is not None:
            # Create inverse permutation
            inv_perm = torch.argsort(self.output_perm)
            out = out[..., inv_perm]

        return out

    @property
    def num_params(self):
        # Permutations don't count as trainable params
        return self.tensorized_linear.num_params

    def to_matrix(self):
        """Reconstruct permuted weight matrix."""
        W_perm = self.tensorized_linear.to_matrix()

        # Reverse permutations
        inv_input_perm = torch.argsort(self.input_perm)
        W = W_perm[:, inv_input_perm]

        if self.output_perm is not None:
            inv_output_perm = torch.argsort(self.output_perm)
            W = W[inv_output_perm, :]

        return W


__all__ = ['PermutedTensorizedLinear', 'compute_feature_correlations', 'spectral_reordering']
