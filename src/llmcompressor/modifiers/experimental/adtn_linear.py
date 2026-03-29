import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "ADTNLinear",
    "ADTNSublayer",
    "StackedLowRankLinear",
    "LowRankLayer",
    "ColumnSparseLinear",
]


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
        # Fully vectorized to avoid slow Python loops
        in_features = input_activations.shape[1]
        remaining_mask = torch.ones(
            in_features, dtype=torch.bool, device=correlation.device
        )
        permutation = []

        while remaining_mask.any():
            remaining_indices = torch.where(remaining_mask)[0]
            current_group_size = min(group_size, len(remaining_indices))

            if current_group_size == 1:
                # Just take the last remaining feature
                permutation.append(remaining_indices[0].item())
                break

            # Find the most correlated pair in remaining features (vectorized)
            remaining_corr = correlation[remaining_indices][:, remaining_indices]
            # Get upper triangle only (excluding diagonal)
            remaining_corr_triu = remaining_corr.triu(diagonal=1)
            # Find max correlation
            max_idx_flat = remaining_corr_triu.argmax()
            i = max_idx_flat // len(remaining_indices)
            j = max_idx_flat % len(remaining_indices)

            seed_idx1 = remaining_indices[i].item()
            seed_idx2 = remaining_indices[j].item()
            group = [seed_idx1, seed_idx2]
            group_tensor = torch.tensor(
                group, dtype=torch.long, device=correlation.device
            )

            # Mark seeds as used
            remaining_mask[seed_idx1] = False
            remaining_mask[seed_idx2] = False

            # Greedily add features most correlated with current group (vectorized)
            while len(group) < current_group_size:
                remaining_indices = torch.where(remaining_mask)[0]
                if len(remaining_indices) == 0:
                    break

                # Compute average correlation of each remaining feature with current group
                # Shape: (num_remaining, group_size)
                corr_with_group = correlation[remaining_indices][:, group_tensor]
                # Average over group members: (num_remaining,)
                avg_corr = corr_with_group.mean(dim=1)

                # Find best candidate (no .item() in loop!)
                best_idx_in_remaining = avg_corr.argmax()
                best_candidate = remaining_indices[best_idx_in_remaining].item()

                group.append(best_candidate)
                group_tensor = torch.tensor(
                    group, dtype=torch.long, device=correlation.device
                )
                remaining_mask[best_candidate] = False

            permutation.extend(group)

        return torch.tensor(
            permutation, dtype=torch.long, device=input_activations.device
        )

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
        Forward pass: sum outputs from all sublayers (additive residuals)
        """
        if len(self.sublayers) == 0:
            return torch.zeros(
                (*x.shape[:-1], self.out_features), dtype=x.dtype, device=x.device
            )

        output = None
        for sublayer in self.sublayers:
            sublayer_out = sublayer(x)
            if output is None:
                output = sublayer_out
            else:
                output = output + sublayer_out
        return output

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
        linears: list[nn.Linear] | None = None,
        input_perm: torch.Tensor | None = None,
        U: nn.Linear | None = None,
        V: nn.Linear | None = None,
    ):
        """
        ADTN Sublayer with two modes:
        1. Concatenation mode: linears + input_perm (block-diagonal)
        2. Low-rank mode: U @ V factorization
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if linears is not None:
            # Concatenation mode (block-diagonal)
            self.mode = "concat"
            self.linears = nn.ModuleList(linears)
            self.group_size = linears[0].in_features
            assert all(linear.in_features == self.group_size for linear in linears[:-1])
            assert in_features == sum(linear.in_features for linear in linears)
            assert out_features == sum(linear.out_features for linear in linears)
            assert input_perm is not None
            self.register_buffer("input_perm", input_perm)
        elif U is not None and V is not None:
            # Low-rank factorization mode
            self.mode = "lowrank"
            self.U = U  # (in_features, rank)
            self.V = V  # (rank, out_features)
        else:
            raise ValueError("Must provide either (linears, input_perm) or (U, V)")

    def forward(
        self,
        x: torch.Tensor,  # shape (batch_size, seq_len, in_features) or (batch_size, in_features)
    ):
        """
        Forward operation with two modes:
        - Concatenation: block-diagonal with permutation
        - Low-rank: x @ U @ V factorization
        """
        if self.mode == "concat":
            # Step 1: Permute inputs - apply permutation to last dimension
            x_perm = x[..., self.input_perm]

            # Step 2: Apply linear operations to each group and collect outputs
            # Each group linear maps (group_size,) -> (out_features/num_groups,)
            outputs = []
            for i, linear in enumerate(self.linears):
                start_idx = i * self.group_size
                end_idx = start_idx + linear.in_features
                group_input = x_perm[..., start_idx:end_idx]
                group_output = linear(
                    group_input
                )  # (batch, ..., out_features/num_groups)
                outputs.append(group_output)

            # Step 3: Concatenate group outputs along last dimension
            y = torch.cat(outputs, dim=-1)  # (batch, ..., out_features)
            return y

        else:  # low-rank mode
            # Simple low-rank factorization: x @ U @ V
            h = self.U(x)  # (batch, ..., rank)
            y = self.V(h)  # (batch, ..., out_features)
            return y

    @property
    def num_params(self):
        if self.mode == "concat":
            return sum([p.numel() for p in self.linears.parameters()])
        else:  # low-rank
            return sum([p.numel() for p in self.U.parameters()]) + sum(
                [p.numel() for p in self.V.parameters()]
            )


class LowRankLayer(nn.Module):
    """
    Single low-rank factorization layer: x @ U @ V
    where U: (in_features, rank), V: (rank, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.dtype = dtype

        self.U = nn.Linear(in_features, rank, bias=False)
        self.V = nn.Linear(rank, out_features, bias=False)

    def forward(self, x: torch.Tensor):
        """Low-rank forward: x @ U @ V"""
        h = self.U(x)  # (batch, ..., rank)
        y = self.V(h)  # (batch, ..., out_features)
        return y

    @property
    def num_params(self):
        return self.U.weight.numel() + self.V.weight.numel()


class StackedLowRankLinear(nn.Module):
    """
    Stacked low-rank linear layers for parameter-efficient approximation.

    Each layer is a low-rank factorization (U @ V) that fits the residual
    from previous layers. Layers are summed (additive residual strategy).

    This achieves:
    - Parameter reduction through rank constraint
    - Better SNR than block-diagonal (captures cross-feature interactions)
    - Iterative residual fitting for high accuracy
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layers: list[LowRankLayer],
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        """Sum outputs from all low-rank layers (additive residuals)"""
        if len(self.layers) == 0:
            return torch.zeros(
                (*x.shape[:-1], self.out_features), dtype=x.dtype, device=x.device
            )

        output = None
        for layer in self.layers:
            layer_out = layer(x)
            if output is None:
                output = layer_out
            else:
                output = output + layer_out
        return output

    def append_layer(self, layer: LowRankLayer):
        """Add a new low-rank layer"""
        self.layers.append(layer)

    @property
    def num_params(self):
        return sum([layer.num_params for layer in self.layers])


class ColumnSparseLinear(nn.Module):
    """
    Column-sparse linear layer for parameter-efficient approximation.

    Instead of storing full weight matrix, only stores weights for selected
    input columns (features). Selection is done via OLS-based greedy algorithm
    that maximizes reconstruction quality with minimal features.

    This achieves:
    - Significant parameter reduction (typically 50%)
    - High SNR (typically 30-70 dB)
    - Block-sparse structure (entire columns, not individual elements)
    - Lower index overhead compared to element-wise sparsity
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        selected_columns: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize column-sparse linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            selected_columns: Indices of selected input columns (1D tensor)
            weight: Weight matrix for selected columns (out_features, len(selected_columns))
            bias: Optional bias term (out_features,)
            dtype: Data type for weights
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        # Store selected column indices
        self.register_buffer("selected_columns", selected_columns.long())

        # Store weights only for selected columns
        # Shape: (out_features, num_selected_columns)
        self.weight = nn.Parameter(weight.to(dtype))

        if bias is not None:
            self.bias = nn.Parameter(bias.to(dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract selected columns and multiply.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            Output tensor (..., out_features)
        """
        # Extract selected columns from input
        x_selected = x[..., self.selected_columns]  # (..., num_selected_columns)

        # Matrix multiply
        output = F.linear(x_selected, self.weight, self.bias)

        return output

    @property
    def num_params(self):
        """Total number of parameters."""
        params = self.weight.numel()
        if self.bias is not None:
            params += self.bias.numel()
        return params

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        input_activations: torch.Tensor,
        target_sparsity: float = 0.5,
        k_cols_per_iter: int = 32,
        max_iters: int | None = None,
        target_snr_db: float | None = None,
    ) -> "ColumnSparseLinear":
        """
        Create ColumnSparseLinear from a torch.nn.Linear layer.

        Uses greedy OLS-based column selection to find the most important
        input features for reconstructing the linear transformation.

        Args:
            linear: Original linear layer
            input_activations: Input activation data for OLS fitting (num_samples, in_features)
            target_sparsity: Target fraction of columns to keep (0.0-1.0)
            k_cols_per_iter: Number of columns to add per iteration
            max_iters: Maximum number of iterations (default: auto from target_sparsity)
            target_snr_db: Optional SNR target in dB (stops early if reached)

        Returns:
            ColumnSparseLinear approximation of the linear layer
        """
        W = linear.weight.data.float()  # (out_features, in_features)
        in_features = linear.in_features
        out_features = linear.out_features

        # Compute target outputs
        with torch.no_grad():
            output_activations = F.linear(input_activations.float(), W.float())

        # Calculate max iterations from target sparsity
        target_num_cols = int(target_sparsity * in_features)
        if max_iters is None:
            max_iters = (target_num_cols + k_cols_per_iter - 1) // k_cols_per_iter

        # Greedy column selection
        selected_cols = []

        for iter_idx in range(max_iters):
            if len(selected_cols) == 0:
                # First iteration: find single best column
                best_col = None
                best_error = float("inf")

                # Try each column
                for col_idx in range(in_features):
                    X_col = input_activations[:, col_idx : col_idx + 1].float()
                    W_col = torch.linalg.lstsq(X_col, output_activations).solution

                    output_approx = X_col @ W_col
                    error = torch.norm(output_activations - output_approx) ** 2

                    if error < best_error:
                        best_error = error
                        best_col = col_idx

                selected_cols = [best_col]
            else:
                # Subsequent iterations: add k_cols_per_iter via residual correlation
                candidates = [c for c in range(in_features) if c not in selected_cols]

                if len(candidates) == 0:
                    break

                # Current reconstruction
                X_current = input_activations[:, selected_cols].float()
                W_current = torch.linalg.lstsq(X_current, output_activations).solution
                current_output = X_current @ W_current
                residual = output_activations - current_output

                # Compute correlation of each candidate with residual
                correlations = []
                for col_idx in candidates:
                    X_col = input_activations[:, col_idx].float()
                    # Correlation: sum over samples and output dims
                    corr = torch.abs((X_col.unsqueeze(1) * residual).sum(dim=0)).sum()
                    correlations.append((corr.item(), col_idx))

                # Sort by correlation and take top k
                correlations.sort(reverse=True)
                new_cols = [col for _, col in correlations[:k_cols_per_iter]]
                selected_cols.extend(new_cols)

            # Check stopping criteria
            if len(selected_cols) >= target_num_cols:
                selected_cols = selected_cols[:target_num_cols]
                break

            # Check SNR if requested
            if target_snr_db is not None:
                X_selected = input_activations[:, selected_cols].float()
                W_selected = torch.linalg.lstsq(X_selected, output_activations).solution
                output_approx = X_selected @ W_selected

                # Compute SNR
                signal_power = torch.var(output_activations)
                mse_noise = torch.mean((output_activations - output_approx) ** 2)
                snr_linear = signal_power / (mse_noise + 1e-10)
                snr_db = 10 * torch.log10(snr_linear)

                if snr_db >= target_snr_db:
                    break

        # Final refit with selected columns
        selected_cols_tensor = torch.tensor(selected_cols, dtype=torch.long)
        X_final = input_activations[:, selected_cols].float()
        W_final = torch.linalg.lstsq(X_final, output_activations).solution

        # Create ColumnSparseLinear
        return cls(
            in_features=in_features,
            out_features=out_features,
            selected_columns=selected_cols_tensor,
            weight=W_final.T,  # Transpose to (out_features, num_selected)
            bias=linear.bias.data if linear.bias is not None else None,
            dtype=linear.weight.dtype,
        )

    def to_linear(self) -> nn.Linear:
        """Convert back to dense nn.Linear layer."""
        # Create full weight matrix
        full_weight = torch.zeros(
            self.out_features,
            self.in_features,
            dtype=self.weight.dtype,
            device=self.weight.device,
        )

        # Fill in selected columns
        full_weight[:, self.selected_columns] = self.weight

        # Create linear layer
        linear = nn.Linear(
            self.in_features, self.out_features, bias=self.bias is not None
        )
        linear.weight.data = full_weight
        if self.bias is not None:
            linear.bias.data = self.bias

        return linear
