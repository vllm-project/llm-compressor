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
    ) -> torch.Tensor:
        """
        Compute optimal channel permutation using Spectral Reordering.
        Same as TensorizedLinear implementation.

        TODO make iterative -- find group_size most correlated to add to permutation,
            then find next group_size most correlated from remaining inputs, and so on.
        """
        activations_centered = input_activations - input_activations.mean(
            dim=0, keepdim=True
        )
        activations_std = activations_centered.std(dim=0, keepdim=True) + 1e-10
        activations_normalized = activations_centered / activations_std
        num_samples = activations_normalized.shape[0]
        correlation = (activations_normalized.T @ activations_normalized) / num_samples

        affinity = torch.exp(correlation - 1.0)
        degree = torch.diag(affinity.sum(dim=1))
        laplacian = degree - affinity

        laplacian_f32 = laplacian.to(torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_f32)
        fiedler_vector = eigenvectors[:, 1]
        input_perm = torch.argsort(fiedler_vector)

        return input_perm.to(input_activations.device)

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
        assert all(linear.group_size == self.group_size for linear in linears)
        assert in_features == self.group_size * len(self.linears)
        self.input_perm = input_perm
        self.input_inv_perm = torch.argsort(input_perm.detach())

    def forward(
        self,
        x: torch.Tensor,  # shape (batch_size, seq_len, in_features)
    ):
        """
        Forward operation:
        1. permute inputs
        2. apply linear operations to each group of input activations
        3. un-permute outputs
        """
        xperm = self.input_perm[x]

        # TODO vectorized apply of linears
        yperm = xperm

        return self.input_inv_perm[yperm]  # shape (batch_size, seq_len, out_features)

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])
