import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ADTNLinear", "ADTNSublayer"]


class ADTNLinear(nn.Module):
    """
    Automatically Differentiable Tensor Network (ADTN) Linear layer.

    ADTNLinear provides a way to perform a torch.nn.Linear operation as a series of
    smaller linear operations on groups of activations that are most correlated, i.e.
    convert a 1000x1000 matmul into 10 100x100 matmuls in each sublayer if group_size is
    100. If the num_sublayers<<10 and the forward pass retains sufficient signal-to-
    noise ratio, this tensorized operation can reduce the memory and number of floating
    point operations compared to the original matmul, though compute-bound runtime
    speed will likely be smaller due to its sequential nature.

    Inspired by: "Compressing Neural Networks Using Tensor Networks with Exponentially
        Fewer Variational Parameters" (https://arxiv.org/pdf/2305.06058)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sublayers: list["ADTNSublayer"],
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.dtype = dtype

        self.sublayers = nn.ModuleList(sublayers)

    @staticmethod
    def _spectral_reordering(
        input_activations: torch.Tensor,
        group_size: int | None = None,
    ) -> torch.Tensor:
        """
        Compute optimal channel permutation.

        If group_size is provided, iteratively finds groups of most correlated features.
        Otherwise, uses global spectral reordering (Fiedler vector).
        """
        # Compute correlation matrix
        activations_centered = input_activations - input_activations.mean(
            dim=0, keepdim=True
        )
        activations_std = activations_centered.std(dim=0, keepdim=True) + 1e-10
        activations_normalized = activations_centered / activations_std
        num_samples = activations_normalized.shape[0]
        correlation = (activations_normalized.T @ activations_normalized) / num_samples

        # If no group_size specified, use global spectral reordering
        if group_size is None:
            affinity = torch.exp(correlation - 1.0)
            degree = torch.diag(affinity.sum(dim=1))
            laplacian = degree - affinity

            laplacian_f32 = laplacian.to(torch.float32)
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_f32)
            fiedler_vector = eigenvectors[:, 1]
            input_perm = torch.argsort(fiedler_vector)

            return input_perm.to(input_activations.device)

        # Iterative grouping: find group_size most correlated, then next group_size, etc.
        in_features = input_activations.shape[1]
        remaining = set(range(in_features))
        permutation = []

        while remaining:
            current_group_size = min(group_size, len(remaining))

            # Find the most correlated pair in remaining features as seed
            best_pair = None
            best_corr = -float('inf')
            remaining_list = list(remaining)

            for i in range(len(remaining_list)):
                for j in range(i + 1, len(remaining_list)):
                    idx_i, idx_j = remaining_list[i], remaining_list[j]
                    corr = correlation[idx_i, idx_j].item()
                    if corr > best_corr:
                        best_corr = corr
                        best_pair = (idx_i, idx_j)

            # Start group with the most correlated pair (or single feature if only one left)
            if best_pair and current_group_size >= 2:
                group = list(best_pair)
                available = remaining - set(best_pair)
            elif remaining:
                group = [remaining_list[0]]
                available = remaining - {remaining_list[0]}
            else:
                break

            # Greedily add features most correlated with current group
            while len(group) < current_group_size and available:
                best_candidate = None
                best_score = -float('inf')

                for candidate in available:
                    # Average correlation with group members
                    score = sum(correlation[candidate, member].item() for member in group) / len(group)
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate

                if best_candidate is not None:
                    group.append(best_candidate)
                    available.remove(best_candidate)
                else:
                    break

            permutation.extend(group)
            remaining -= set(group)

        return torch.tensor(permutation, dtype=torch.long, device=input_activations.device)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: str | float | int = 0.5,
        num_sites: int = 8,
        num_layers: int = 3,
        input_activations: torch.Tensor | None = None,
    ) -> "ADTNLinear":
        """
        Build ADTNLinear from a torch.nn.Linear layer.

        Args:
            linear: Original linear layer
            rank: Bond dimension for internal connections
                  If float, determines compression ratio
                  If int, uses that rank directly
            num_sites: Number of sites/tensors in the brick-wall circuit
            num_layers: Number of alternating layers
            input_activations: Optional activation data for spectral reordering
        """
        return cls.from_weight_and_bias(
            linear.weight.detach(),
            linear.bias.detach() if linear.bias is not None else None,
            rank,
            num_sites,
            num_layers,
            input_activations,
        )

    def append_sublayer(self, sublayer: "ADTNLinear"):
        self.sublayers.append(sublayer)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through each sublayer
        """
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x

    def dense_forward(self, x):
        """Forward pass using dense weight matrix reconstruction."""
        linear = self.to_linear()
        return linear(x)

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])

    def to_linear(self) -> nn.Linear:
        """Convert back to dense nn.Linear layer."""
        pass


class ADTNSublayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        linears: list[nn.Linear],
        input_perm: torch.Tensor,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linears = nn.ModuleList(linears)
        self.group_size = linears[0].in_features
        assert all(linear.in_features == self.group_size for linear in linears[:-1])
        assert in_features == sum(linear.in_features for linear in linears)
        self.register_buffer("input_perm", input_perm)

    def forward(
        self,
        x: torch.Tensor,  # shape (batch_size, seq_len, in_features) or (batch_size, in_features)
    ):
        """
        Forward operation:
        1. permute inputs so correlated features are grouped together
        2. apply linear operations to each group of input activations
        3. sum outputs from all groups (each group produces full output_features)
        """
        # Step 1: Permute inputs - apply permutation to last dimension
        x_perm = x[..., self.input_perm]

        # Step 2: Apply linear operations to each group and sum their outputs
        # Each group linear maps (group_size,) -> (out_features,)
        y = None
        for i, linear in enumerate(self.linears):
            start_idx = i * self.group_size
            end_idx = start_idx + linear.in_features
            group_input = x_perm[..., start_idx:end_idx]
            group_output = linear(group_input)

            if y is None:
                y = group_output
            else:
                y = y + group_output

        return y  # shape (batch_size, seq_len, out_features) or (batch_size, out_features)

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])
