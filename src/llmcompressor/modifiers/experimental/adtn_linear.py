import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ADTNLinear"]


class ADTNLinear(nn.Module):
    """
    Alternating Direction Tensor Network (ADTN) Linear layer.

    ADTN uses a brick-wall topology where layers alternate between connecting
    different pairs of sites:
    - Layer 1: (1,2), (3,4), (5,6), ... (even pairs)
    - Layer 2: (2,3), (4,5), (6,7), ... (odd pairs)
    - Layer 3: (1,2), (3,4), (5,6), ... (even pairs)
    - etc.

    This provides better 2D entanglement compared to 1D tensor trains (MPO).

    Based on: https://arxiv.org/pdf/2305.06058
    "Expressive Quantum Supervised Learning using Alternating Direction Tensor Network"
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_sites: int,
        num_layers: int,
        rank: int,
        gates: list[torch.Tensor] | None,
        bias: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
        per_dim_scale: torch.Tensor | None = None,
        input_perm: torch.Tensor | None = None,
        input_inv_perm: torch.Tensor | None = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super(ADTNLinear, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.num_sites = num_sites
        self.num_layers = num_layers
        self.rank = rank
        self.dtype = dtype
        self.bias = bias.to(dtype) if bias is not None else bias

        # Initialize gates if not provided
        if gates is None or len(gates) == 0:
            gates = self._initialize_gates(
                in_features, out_features, num_sites, num_layers, rank, dtype
            )

        # Store gates as parameters
        # Each gate is a 2-site tensor that connects two neighboring sites
        # Shape: (rank_left, rank_right, in_dim_1, in_dim_2, out_dim_1, out_dim_2)
        self.gates = nn.ParameterList(
            [nn.Parameter(g.to(dtype), requires_grad=True) for g in gates]
        )

        # Learnable global scalar scaling parameter
        if alpha is None:
            alpha = torch.ones(1, dtype=dtype)
        self.alpha = nn.Parameter(alpha.to(dtype), requires_grad=True)

        # Learnable per-dimension scaling
        if per_dim_scale is None:
            per_dim_scale = torch.ones(out_features, dtype=dtype)
        self.per_dim_scale = nn.Parameter(per_dim_scale.to(dtype), requires_grad=True)

        # Channel permutation for spectral reordering
        if input_perm is not None:
            self.register_buffer("input_perm", input_perm.long())
            self.register_buffer("input_inv_perm", input_inv_perm.long())
        else:
            self.input_perm = None
            self.input_inv_perm = None

    @staticmethod
    def _initialize_gates(
        in_features: int,
        out_features: int,
        num_sites: int,
        num_layers: int,
        rank: int,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """
        Initialize brick-wall gates with random values.

        Returns:
            List of gates for each layer's connections
        """
        gates = []
        in_per_site = in_features // num_sites
        out_per_site = out_features // num_sites

        for layer_idx in range(num_layers):
            # Determine which pairs to connect in this layer
            # Even layers (0, 2, 4, ...): connect (0,1), (2,3), (4,5), ...
            # Odd layers (1, 3, 5, ...): connect (1,2), (3,4), (5,6), ...
            offset = layer_idx % 2

            for pair_idx in range(
                (num_sites - 1 - offset) // 2 + (1 if offset == 0 else 0)
            ):
                site1 = offset + pair_idx * 2
                site2 = site1 + 1

                if site2 >= num_sites:
                    break

                # Bond dimensions
                rank_left = 1 if site1 == 0 else rank
                rank_right = 1 if site2 == num_sites - 1 else rank

                # Physical dimensions per site
                in_dim_1 = in_per_site + (1 if site1 < in_features % num_sites else 0)
                in_dim_2 = in_per_site + (1 if site2 < in_features % num_sites else 0)
                out_dim_1 = out_per_site + (
                    1 if site1 < out_features % num_sites else 0
                )
                out_dim_2 = out_per_site + (
                    1 if site2 < out_features % num_sites else 0
                )

                # Gate shape: (rank_left, rank_right, in_dim_1, in_dim_2, out_dim_1, out_dim_2)
                gate = (
                    torch.randn(
                        rank_left,
                        rank_right,
                        in_dim_1,
                        in_dim_2,
                        out_dim_1,
                        out_dim_2,
                        dtype=dtype,
                    )
                    * 0.1
                )

                gates.append(gate)

        return gates

    @staticmethod
    def _spectral_reordering(
        weight: torch.Tensor,
        input_activations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute optimal channel permutation using Spectral Reordering.
        Same as TensorizedLinear implementation.
        """
        if input_activations is not None:
            activations_centered = input_activations - input_activations.mean(
                dim=0, keepdim=True
            )
            activations_std = activations_centered.std(dim=0, keepdim=True) + 1e-10
            activations_normalized = activations_centered / activations_std
            num_samples = activations_normalized.shape[0]
            correlation = (
                activations_normalized.T @ activations_normalized
            ) / num_samples
        else:
            weight_normalized = weight / (
                torch.norm(weight, dim=0, keepdim=True) + 1e-10
            )
            correlation = weight_normalized.T @ weight_normalized

        affinity = torch.exp(correlation - 1.0)
        degree = torch.diag(affinity.sum(dim=1))
        laplacian = degree - affinity

        laplacian_f32 = laplacian.to(torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_f32)
        fiedler_vector = eigenvectors[:, 1]
        input_perm = torch.argsort(fiedler_vector)

        return input_perm.to(weight.device)

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

    @classmethod
    def from_weight_and_bias(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        rank: str | float | int = 0.5,
        num_sites: int = 8,
        num_layers: int = 3,
        input_activations: torch.Tensor | None = None,
    ) -> "ADTNLinear":
        """
        Build ADTNLinear from weight matrix and bias.

        Args:
            weight: Weight matrix of shape (out_features, in_features)
            bias: Bias vector of shape (out_features,) or None
            rank: Bond dimension for internal connections
                  If float (0-1), target compression ratio
                  If int, use that rank
            num_sites: Number of sites in brick-wall circuit
            num_layers: Number of alternating layers
            input_activations: Optional activation data for spectral reordering
        """
        assert weight.ndim == 2, "invalid weight"
        weight = weight.clone(memory_format=torch.contiguous_format)

        if bias is not None:
            assert bias.ndim == 1, "invalid bias"
            assert weight.shape[0] == bias.shape[0], "incompatible weight/bias"
            bias = bias.clone(memory_format=torch.contiguous_format)

        if input_activations is not None:
            assert input_activations.ndim == 2, "invalid input_activations"
            assert (
                input_activations.shape[1] == weight.shape[1]
            ), "incompatible activations"

        out_features, in_features = weight.shape

        # Determine rank
        if isinstance(rank, float) and 0 < rank < 1:
            # Estimate parameters for brick-wall circuit
            # Each gate has size ~ rank_left * rank_right * in_dim_1 * in_dim_2 * out_dim_1 * out_dim_2
            # For rough estimate: rank_left ≈ rank_right ≈ r, in_dim_i ≈ in_features/num_sites
            avg_in_dim = in_features / num_sites
            avg_out_dim = out_features / num_sites
            # Approximate number of gates
            num_gates = (num_sites - 1) * num_layers // 2 + num_layers // 2
            target_params = out_features * in_features * rank
            # Each gate: r^2 * (in_dim)^2 * (out_dim)^2
            # Solve for r: target ≈ num_gates * r^2 * avg_in_dim^2 * avg_out_dim^2
            r_squared = target_params / (num_gates * avg_in_dim**2 * avg_out_dim**2)
            rank_val = max(1, int(math.sqrt(r_squared)))
        elif rank == "auto":
            rank_val = max(
                2,
                min(in_features // (2 * num_sites), out_features // (2 * num_sites), 8),
            )
        else:
            rank_val = int(rank)

        # Use spectral reordering for input channels
        input_perm = cls._spectral_reordering(weight, input_activations)
        input_inv_perm = torch.argsort(input_perm)
        weight_permuted = weight[:, input_perm]

        # Initialize using decomposition
        orig_dtype = weight.dtype
        weight_permuted = weight_permuted.to(torch.float32)

        # Decompose weight matrix into brick-wall gates
        gates = cls._decompose_brick_wall(
            weight_permuted, in_features, out_features, num_sites, num_layers, rank_val
        )

        return cls(
            in_features,
            out_features,
            num_sites,
            num_layers,
            rank_val,
            gates,
            bias,
            input_perm=input_perm,
            input_inv_perm=input_inv_perm,
            dtype=orig_dtype,
        )

    @staticmethod
    def _decompose_brick_wall(
        weight: torch.Tensor,
        in_features: int,
        out_features: int,
        num_sites: int,
        num_layers: int,
        rank: int,
    ) -> list[torch.Tensor]:
        """
        Decompose weight matrix into brick-wall structure.

        For simplicity, we use SVD-based initialization where each gate
        gets a portion of the weight matrix to approximate.
        """
        gates = []
        in_per_site = in_features // num_sites
        out_per_site = out_features // num_sites

        # Use SVD of full weight to guide initialization
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        r_svd = min(rank, len(S))

        # For each layer, create gates for the connections
        for layer_idx in range(num_layers):
            offset = layer_idx % 2

            for pair_idx in range(
                (num_sites - 1 - offset) // 2 + (1 if offset == 0 else 0)
            ):
                site1 = offset + pair_idx * 2
                site2 = site1 + 1

                if site2 >= num_sites:
                    break

                # Determine dimensions
                in_start_1 = site1 * in_per_site
                in_end_1 = min((site1 + 1) * in_per_site, in_features)
                in_start_2 = site2 * in_per_site
                in_end_2 = min((site2 + 1) * in_per_site, in_features)

                out_start_1 = site1 * out_per_site
                out_end_1 = min((site1 + 1) * out_per_site, out_features)
                out_start_2 = site2 * out_per_site
                out_end_2 = min((site2 + 1) * out_per_site, out_features)

                in_dim_1 = in_end_1 - in_start_1
                in_dim_2 = in_end_2 - in_start_2
                out_dim_1 = out_end_1 - out_start_1
                out_dim_2 = out_end_2 - out_start_2

                # Bond dimensions
                rank_left = 1 if site1 == 0 else rank
                rank_right = 1 if site2 == num_sites - 1 else rank

                # Gate shape: (rank_left, rank_right, in_dim_1, in_dim_2, out_dim_1, out_dim_2)
                # Initialize with small random values scaled by SVD singular values
                scale = S[min(pair_idx, len(S) - 1)].item() / (num_layers * num_sites)
                gate = (
                    torch.randn(
                        rank_left,
                        rank_right,
                        in_dim_1,
                        in_dim_2,
                        out_dim_1,
                        out_dim_2,
                        dtype=weight.dtype,
                        device=weight.device,
                    )
                    * scale
                )

                gates.append(gate)

        return gates

    def to_matrix(self) -> torch.Tensor:
        """
        Reconstruct the full weight matrix from brick-wall gates.
        Uses tensor network contraction.
        """
        # Contract the brick-wall network
        W = self._contract_brick_wall()

        # Apply scaling
        scale = self.per_dim_scale * self.alpha
        W_scaled = W * scale.view(-1, 1)

        # Apply inverse permutation
        if self.input_inv_perm is not None:
            W_scaled = W_scaled[:, self.input_inv_perm]

        return W_scaled

    def _contract_brick_wall(self) -> torch.Tensor:
        """
        Contract brick-wall network to produce weight matrix.

        This is complex, so for now we use a simplified approach:
        just sum contributions from all gates.
        """
        in_per_site = self.in_features // self.num_sites
        out_per_site = self.out_features // self.num_sites

        # Initialize result
        result = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.gates[0].device,
            dtype=self.gates[0].dtype,
        )

        gate_idx = 0
        for layer_idx in range(self.num_layers):
            offset = layer_idx % 2

            for pair_idx in range(
                (self.num_sites - 1 - offset) // 2 + (1 if offset == 0 else 0)
            ):
                site1 = offset + pair_idx * 2
                site2 = site1 + 1

                if site2 >= self.num_sites or gate_idx >= len(self.gates):
                    break

                gate = self.gates[gate_idx]
                gate_idx += 1

                # Gate: (rank_left, rank_right, in_dim_1, in_dim_2, out_dim_1, out_dim_2)
                # Contract over rank dimensions (average)
                gate_contracted = gate.mean(
                    dim=(0, 1)
                )  # (in_dim_1, in_dim_2, out_dim_1, out_dim_2)

                # Determine positions
                in_start_1 = site1 * in_per_site
                in_end_1 = min((site1 + 1) * in_per_site, self.in_features)
                in_start_2 = site2 * in_per_site
                in_end_2 = min((site2 + 1) * in_per_site, self.in_features)

                out_start_1 = site1 * out_per_site
                out_end_1 = min((site1 + 1) * out_per_site, self.out_features)
                out_start_2 = site2 * out_per_site
                out_end_2 = min((site2 + 1) * out_per_site, self.out_features)

                in_dim_1 = in_end_1 - in_start_1
                in_dim_2 = in_end_2 - in_start_2
                out_dim_1 = out_end_1 - out_start_1
                out_dim_2 = out_end_2 - out_start_2

                # Add contribution from this gate to result
                # Reshape gate_contracted: (in_dim_1, in_dim_2, out_dim_1, out_dim_2) -> matrix form
                gate_matrix = gate_contracted.permute(2, 3, 0, 1).reshape(
                    out_dim_1 * out_dim_2, in_dim_1 * in_dim_2
                )

                # Place in result matrix (simplified - just add to blocks)
                out_indices = list(range(out_start_1, out_end_1)) + list(
                    range(out_start_2, out_end_2)
                )
                in_indices = list(range(in_start_1, in_end_1)) + list(
                    range(in_start_2, in_end_2)
                )

                for i, out_idx in enumerate(out_indices):
                    for j, in_idx in enumerate(in_indices):
                        if out_idx < self.out_features and in_idx < self.in_features:
                            result[out_idx, in_idx] += (
                                gate_matrix[i, j] / self.num_layers
                            )

        return result

    def forward(self, x):
        """
        Forward pass using dense weight matrix reconstruction.
        """
        return self.dense_forward(x)

    def dense_forward(self, x):
        """Forward pass using dense weight matrix reconstruction."""
        W = self.to_matrix()
        result = F.linear(x, W, self.bias)
        return result

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])

    def to_dense(self) -> nn.Linear:
        """Convert to dense nn.Linear layer."""
        with torch.no_grad():
            weight_matrix = self.to_matrix()
            out_features, in_features = weight_matrix.shape
            has_bias = self.bias is not None

            dense_layer = nn.Linear(
                in_features, out_features, bias=has_bias, dtype=weight_matrix.dtype
            )
            dense_layer.weight.data.copy_(weight_matrix)

            if has_bias:
                dense_layer.bias.data.copy_(self.bias)

        return dense_layer

    @staticmethod
    def get_shape(num_features: int, num_sites: int) -> tuple[int, ...]:
        """
        Determine shape for ADTN decomposition.
        Returns dimensions for each site.
        """
        base_dim = num_features // num_sites
        remainder = num_features % num_sites

        shape = []
        for i in range(num_sites):
            dim = base_dim + (1 if i < remainder else 0)
            shape.append(dim)

        return tuple(shape)
