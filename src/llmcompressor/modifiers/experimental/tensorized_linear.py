import math

import tensorly as tl
import torch
import torch.nn.functional as F
from tensorly.tt_matrix import TTMatrix
from torch import nn

tl.set_backend("pytorch")

__all__ = ["TensorizedLinear"]


# TODO move next to compressed_tensors..CompressedLinear
class TensorizedLinear(nn.Module):
    """
    Stores the weights of a fully connected layer in the TT-matrix format

    Original implemtntation:
    https://github.com/tensorly/Proceedings_IEEE_companion_notebooks/blob/master/tt-compression.ipynb
    """

    def __init__(
        self,
        shape: tuple[int],
        rank: int | tuple[int] | None,
        factors: tuple[torch.Tensor, ...] | None,
        bias: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
        per_dim_scale: torch.Tensor | None = None,
        input_perm: torch.Tensor | None = None,
        input_inv_perm: torch.Tensor | None = None,
        output_perm: torch.Tensor | None = None,
        output_inv_perm: torch.Tensor | None = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super(TensorizedLinear, self).__init__(**kwargs)

        self.shape = shape
        self.dtype = dtype
        self.bias = bias.to(dtype) if bias is not None else bias

        # if not provided, initialize tt_matrix and weights
        if factors is None or len(factors) == 0:
            tt_matrix = tl.random.random_tt_matrix(shape, rank=rank)
            for f in tt_matrix.factors:
                f.data.uniform_(-0.1, 0.1)
            factors = tt_matrix.factors

        # Add and register the factors
        self.factors = nn.ParameterList(
            [nn.Parameter(f.to(dtype), requires_grad=True) for f in factors]
        )

        # Learnable global scalar scaling parameter initialized to 1.0
        if alpha is None:
            alpha = torch.ones(1, dtype=dtype)
        self.alpha = nn.Parameter(alpha.to(dtype), requires_grad=True)

        # Learnable per-dimension scaling (like LayerNorm's weight)
        # Initialized to 1.0 to preserve initial behavior
        num_cores = len(factors)
        output_shape = shape[:num_cores]
        out_features = math.prod(output_shape)
        if per_dim_scale is None:
            per_dim_scale = torch.ones(out_features, dtype=dtype)
        self.per_dim_scale = nn.Parameter(per_dim_scale.to(dtype), requires_grad=True)

        # Channel permutation indices for importance-based ordering
        # input_perm: permutation to apply to inputs (sort by importance)
        # input_inv_perm: inverse permutation to restore original input order
        if input_perm is not None:
            self.register_buffer("input_perm", input_perm.long())
            self.register_buffer("input_inv_perm", input_inv_perm.long())
        else:
            self.input_perm = None
            self.input_inv_perm = None

        # output_perm: permutation to apply to outputs (sort by importance)
        # output_inv_perm: inverse permutation to restore original output order
        if output_perm is not None:
            self.register_buffer("output_perm", output_perm.long())
            self.register_buffer("output_inv_perm", output_inv_perm.long())
        else:
            self.output_perm = None
            self.output_inv_perm = None

    @staticmethod
    def _spectral_reordering(
        weight: torch.Tensor,
        input_activations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute optimal channel permutation using Spectral Reordering via Laplacian Eigenmaps.

        This finds the permutation that maximizes local coherence for MPO decomposition by:
        1. Computing correlation matrix between input channels from activations or weights
        2. Building graph Laplacian from correlations
        3. Finding Fiedler vector (eigenvector of 2nd smallest eigenvalue)
        4. Sorting channels by Fiedler vector to group similar channels together

        Args:
            weight: Weight matrix of shape (out_features, in_features)
            input_activations: Optional input activation data of shape (num_samples, in_features).
                             If provided, computes correlations from activation patterns instead of weights.

        Returns:
            Permutation indices that maximize local coherence
        """
        if input_activations is not None:
            # Compute correlation from activation data
            # Center activations (subtract mean) for proper correlation
            activations_centered = input_activations - input_activations.mean(dim=0, keepdim=True)

            # Normalize each channel to unit variance for correlation computation
            activations_std = activations_centered.std(dim=0, keepdim=True) + 1e-10
            activations_normalized = activations_centered / activations_std

            # Correlation matrix: Corr[i,j] = correlation between channel i and j
            # Shape: (in_features, in_features)
            num_samples = activations_normalized.shape[0]
            correlation = (activations_normalized.T @ activations_normalized) / num_samples
        else:
            # Compute correlation matrix between input channels (columns of weight)
            # Normalize each column to unit norm for cosine similarity
            weight_normalized = weight / (torch.norm(weight, dim=0, keepdim=True) + 1e-10)

            # Correlation/affinity matrix: C[i,j] = cosine similarity between channels i and j
            # Shape: (in_features, in_features)
            correlation = weight_normalized.T @ weight_normalized

        # Make affinity matrix non-negative and apply exponential kernel for stronger locality
        # A[i,j] = exp(correlation[i,j]) gives higher weight to similar channels
        affinity = torch.exp(correlation - 1.0)  # Subtract 1 so diagonal ≈ 1 after exp

        # Construct graph Laplacian: L = D - A
        # where D is degree matrix (diagonal with row sums of A)
        degree = torch.diag(affinity.sum(dim=1))
        laplacian = degree - affinity

        # Compute eigendecomposition of Laplacian
        # Use float32 for eigendecomposition (more stable than bfloat16)
        laplacian_f32 = laplacian.to(torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_f32)

        # Fiedler vector: eigenvector corresponding to 2nd smallest eigenvalue
        # This vector provides a 1D embedding that preserves graph structure
        fiedler_vector = eigenvectors[:, 1]  # Index 1 = second smallest eigenvalue

        # Sort channels by Fiedler vector values
        # This groups strongly connected channels together, maximizing local coherence
        input_perm = torch.argsort(fiedler_vector)

        return input_perm.to(weight.device)

    @staticmethod
    def _spectral_reordering_output(
        weight: torch.Tensor,
        output_activations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute optimal output channel permutation using Spectral Reordering via Laplacian Eigenmaps.

        This finds the permutation that maximizes local coherence for output dimensions by:
        1. Computing correlation matrix between output channels from activations or weights
        2. Building graph Laplacian from correlations
        3. Finding Fiedler vector (eigenvector of 2nd smallest eigenvalue)
        4. Sorting channels by Fiedler vector to group similar channels together

        Args:
            weight: Weight matrix of shape (out_features, in_features)
            output_activations: Optional output activation data of shape (num_samples, out_features).
                              If provided, computes correlations from activation patterns instead of weights.

        Returns:
            Permutation indices that maximize local coherence for output dimensions
        """
        if output_activations is not None:
            # Compute correlation from activation data
            # Center activations (subtract mean) for proper correlation
            activations_centered = output_activations - output_activations.mean(dim=0, keepdim=True)

            # Normalize each channel to unit variance for correlation computation
            activations_std = activations_centered.std(dim=0, keepdim=True) + 1e-10
            activations_normalized = activations_centered / activations_std

            # Correlation matrix: Corr[i,j] = correlation between output channel i and j
            # Shape: (out_features, out_features)
            num_samples = activations_normalized.shape[0]
            correlation = (activations_normalized.T @ activations_normalized) / num_samples
        else:
            # Compute correlation matrix between output channels (rows of weight)
            # Normalize each row to unit norm for cosine similarity
            weight_normalized = weight / (torch.norm(weight, dim=1, keepdim=True) + 1e-10)

            # Correlation/affinity matrix: C[i,j] = cosine similarity between output channels i and j
            # Shape: (out_features, out_features)
            correlation = weight_normalized @ weight_normalized.T

        # Make affinity matrix non-negative and apply exponential kernel for stronger locality
        # A[i,j] = exp(correlation[i,j]) gives higher weight to similar channels
        affinity = torch.exp(correlation - 1.0)  # Subtract 1 so diagonal ≈ 1 after exp

        # Construct graph Laplacian: L = D - A
        # where D is degree matrix (diagonal with row sums of A)
        degree = torch.diag(affinity.sum(dim=1))
        laplacian = degree - affinity

        # Compute eigendecomposition of Laplacian
        # Use float32 for eigendecomposition (more stable than bfloat16)
        laplacian_f32 = laplacian.to(torch.float32)
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_f32)

        # Fiedler vector: eigenvector corresponding to 2nd smallest eigenvalue
        # This vector provides a 1D embedding that preserves graph structure
        fiedler_vector = eigenvectors[:, 1]  # Index 1 = second smallest eigenvalue

        # Sort channels by Fiedler vector values
        # This groups strongly connected channels together, maximizing local coherence
        output_perm = torch.argsort(fiedler_vector)

        return output_perm.to(weight.device)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: str | float | int | tuple[int] = 0.5,
        num_cores: int = 2,
        input_activations: torch.Tensor | None = None,
    ) -> "TensorizedLinear":
        """
        Build TensorizedLinear from an input torch.nn.Linear layer


        linear: original linear layer's weight matrix
        rank: determines the number of parameters
            if float, sets the total number of parameters to be
                linear.weight.numel() * rank
            if "same", same number of parameters as original
                (completely reconstructs the linear mapping)
        input_activations: Optional input activation data of shape (num_samples, in_features)
                          for activation-based spectral reordering
        """
        return cls.from_weight_and_bias(
            linear.weight.detach(),
            linear.bias.detach() if linear.bias is not None else None,
            rank,
            num_cores,
            input_activations,
        )

    @classmethod
    def from_weight_and_bias(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        rank: str | float | int | tuple[int] = 0.5,
        num_cores: int = 2,
        input_activations: torch.Tensor | None = None,
    ) -> "TensorizedLinear":
        """
        Build TensorizedLinear from an input torch.nn.Linear layer

        linear: original linear layer's weight matrix
        rank: determines the number of parameters
            if float, sets the total number of parameters to be
                linear.weight.numel() * rank
            if "same", same number of parameters as original
                (completely reconstructs the linear mapping)
        input_activations: Optional input activation data of shape (num_samples, in_features)
                          for activation-based spectral reordering. If provided, channel
                          correlations are computed from activation patterns instead of weights.
        """
        assert weight.ndim == 2, "invalid weight"
        # always create fresh
        weight = weight.clone(memory_format=torch.contiguous_format)

        if bias is not None:
            assert bias.ndim == 1, "invalid bias"
            assert weight.shape[0] == bias.shape[0], "incompatible weight/bias"
            bias = bias.clone(memory_format=torch.contiguous_format)

        # Compute output activations if input activations provided
        output_activations = None
        if input_activations is not None:
            assert input_activations.ndim == 2, "invalid input_activations, expected (num_samples, in_features)"
            assert input_activations.shape[1] == weight.shape[1], (
                f"input_activations feature dim {input_activations.shape[1]} doesn't match "
                f"weight in_features {weight.shape[1]}"
            )
            # Compute output activations: output = weight @ input^T
            # Shape: (num_samples, out_features)
            # Ensure dtype compatibility
            with torch.no_grad():
                input_acts_dtype = input_activations.to(weight.dtype)
                output_activations = (weight @ input_acts_dtype.T).T

        # Use spectral reordering to find optimal input channel permutation
        # This maximizes local coherence for MPO decomposition using Laplacian Eigenmaps
        input_perm = cls._spectral_reordering(weight, input_activations)
        input_inv_perm = torch.argsort(input_perm)

        # Use spectral reordering to find optimal output channel permutation
        output_perm = cls._spectral_reordering_output(weight, output_activations)
        output_inv_perm = torch.argsort(output_perm)

        # Apply both permutations to weight matrix
        # Permute rows (output channels) and columns (input channels)
        weight_permuted = weight[output_perm, :][:, input_perm]

        # Note: bias is kept in original order, will be added after unpermuting outputs

        output_shape = cls.get_shape(weight.shape[0], num_cores)
        input_shape = cls.get_shape(weight.shape[1], num_cores)
        # NOTE: Order is based on what is shown in reference notebook
        # It is probably this because the linear weight matrix has
        #   a shape of (out_features, in_features)
        shape = output_shape + input_shape

        # upconvert to float32 before svd, no bfloat16 implementation
        orig_dtype = weight.dtype
        weight_permuted = weight_permuted.to(torch.float32)
        tt_matrix = tl.decomposition.tensor_train_matrix(
            tl.reshape(weight_permuted, shape),
            rank=rank,
        )
        return cls(
            shape,
            rank,
            tt_matrix.factors,
            bias,  # Keep bias in original order
            input_perm=input_perm,
            input_inv_perm=input_inv_perm,
            output_perm=output_perm,
            output_inv_perm=output_inv_perm,
            dtype=orig_dtype,
        )

    @staticmethod
    def get_shape(num_features: int, num_cores: int) -> tuple[int, ...]:
        """
        Given an input dimension of num_features, and a desired MPO
        with number of cores equal to num_cores, return a tuple of
        length num_cores which has a number of elements equal to
        num_features, ideally as powers of 2 and ideally as close to
        each other in size (to maximize entanglement) as possible.
        """
        shape = []
        remainder = num_features
        for i in reversed(range(num_cores)):
            if i == 0:
                shape.append(round(remainder))
            else:
                dim = get_nearest_power_of_2(
                    remainder ** (1 / (num_cores - len(shape)))
                )
                shape.append(dim)
                remainder = remainder / dim
        assert len(shape) == num_cores, "Something wrong with len(shape)"
        assert (
            math.prod(shape) == num_features
        ), f"Something wrong with num_features, {shape}, {num_features}"
        return shape

    def to_matrix(self) -> torch.Tensor:
        """
        Return tensorized weights expanded into a single weight matrix,
        including learned scaling parameters (alpha and per_dim_scale).
        Returns the matrix in the original channel ordering (applies inverse permutations).
        """
        W = tl.tt_matrix.tt_matrix_to_matrix(self.factors)
        # Apply learned scaling: W_scaled[i, :] = per_dim_scale[i] * alpha * W[i, :]
        scale = self.per_dim_scale * self.alpha
        W_scaled = W * scale.view(-1, 1)

        # Apply inverse permutations to restore original channel ordering
        # First restore input channels (columns)
        if self.input_inv_perm is not None:
            W_scaled = W_scaled[:, self.input_inv_perm]

        # Then restore output channels (rows)
        if self.output_inv_perm is not None:
            W_scaled = W_scaled[self.output_inv_perm, :]

        return W_scaled

    def forward(self, x):
        """
        Performs forward pass on input x by building einsum contraction
        string for TT-matrix vector product.

        Supports arbitrary batch dimensions - linear transformation is applied
        to the last dimension, like nn.Linear.
        """
        # Store original shape and flatten all batch dimensions
        original_shape = x.shape
        in_features = original_shape[-1]
        batch_dims = original_shape[:-1]

        # Flatten all batch dimensions: (..., in_features) -> (batch_total, in_features)
        x = x.reshape(-1, in_features)
        batch_total = x.shape[0]

        # Apply input channel permutation if present
        # This reorders channels by importance before MPO contraction
        if self.input_perm is not None:
            x = x[:, self.input_perm]

        # Get shapes from factors
        input_shape = [f.shape[2] for f in self.factors]
        output_shape = [f.shape[1] for f in self.factors]
        num_cores = len(self.factors)

        # Reshape input to expose individual dimensions: (batch, m_0, m_1, ..., m_{d-1})
        x = x.reshape(batch_total, *input_shape)

        # Build einsum string using mixed-case letters
        # lowercase a-z (26) + uppercase A-Z (26) = 52 total characters
        # Use 'b' for batch, then assign remaining 51 chars to tensor train dims
        #
        # For n cores need: n inputs + n outputs + (n+1) ranks = 3n+1 chars
        # With 51 chars: 3n+1 <= 51 means n <= 16 cores max
        #
        # For >16 cores, fall back to sequential contraction

        if num_cores <= 16:
            # Use fast single einsum for ≤16 cores

            def get_char(i):
                """Get character for index i (skip 'b' for batch):
                0-23: c-z (24 chars, skip a,b)
                24-49: A-Z (26 chars)
                Total: 50 chars available
                """
                if i < 24:
                    return chr(ord('c') + i)  # c-z: 24 chars (indices 0-23)
                else:
                    return chr(ord('A') + (i - 24))  # A-Z: 26 chars (indices 24-49)

            # Assign characters for different indices
            # 'b' = batch
            # Input dims: first num_cores chars
            # Output dims: next num_cores chars
            # Rank dims: next num_cores+1 chars

            input_chars = [get_char(i) for i in range(num_cores)]
            output_chars = [get_char(i + num_cores) for i in range(num_cores)]
            rank_chars = [get_char(i + 2*num_cores) for i in range(num_cores + 1)]

            # Input: batch + input dimensions
            input_subscripts = "b" + "".join(input_chars)

            # Each core: rank_i + output_i + input_i + rank_{i+1}
            core_subscripts = [
                f"{rank_chars[i]}{output_chars[i]}{input_chars[i]}{rank_chars[i+1]}"
                for i in range(num_cores)
            ]

            # Output: batch + output dimensions
            output_subscripts = "b" + "".join(output_chars)

            einsum_string = (
                f"{input_subscripts},{','.join(core_subscripts)}->{output_subscripts}"
            )

            # Contract using einsum
            result = torch.einsum(einsum_string, x, *self.factors)

        else:
            # Sequential contraction for >16 cores
            # Use a simple loop-based contraction

            # Initialize: add dummy leading rank dimension
            result = x.unsqueeze(1)  # (batch, 1, m_0, m_1, ..., m_{d-1})
            batch_size = result.shape[0]

            for i, core in enumerate(self.factors):
                # core: (r_i, n_i, m_i, r_{i+1})
                r_i, n_i, m_i, r_ip1 = core.shape

                # result: (batch, r_i, *out_dims_so_far, *in_dims_remaining)
                # where out_dims_so_far = output_shape[:i], in_dims_remaining = input_shape[i:]

                # Flatten everything for simple contraction
                out_so_far = list(output_shape[:i]) if i > 0 else []
                in_remaining = list(input_shape[i:])

                # Reshape to (batch, r_i, prod(out_so_far), m_i, prod(in_remaining[1:]))
                out_flat = math.prod(out_so_far) if out_so_far else 1
                in_rest_flat = math.prod(in_remaining[1:]) if len(in_remaining) > 1 else 1

                result = result.reshape(batch_size, r_i, out_flat, m_i, in_rest_flat)

                # Contract using simple operations
                # result: (batch, r_i, O, m_i, I)
                # core: (r_i, n_i, m_i, r')
                # Want: (batch, r', O, n_i, I)

                # Permute to group dims for bmm: (batch, O, I, r_i, m_i)
                result = result.permute(0, 2, 4, 1, 3)
                # Reshape: (batch * O * I, r_i, m_i)
                bOI = batch_size * out_flat * in_rest_flat
                result = result.reshape(bOI, r_i, m_i)

                # Reshape core: (r_i, n_i * r', m_i)
                core_reshaped = core.permute(0, 3, 1, 2).reshape(r_i, r_ip1 * n_i, m_i)

                # Batch matrix multiply: (bOI, r_i, m_i) @ (r_i, m_i, n_i*r')^T
                #   = (bOI, r_i, m_i) @ (r_i, n_i*r', m_i).transpose(-2, -1)
                # First: for each sample in bOI, contract over r_i and m_i
                # result[k, :, :] is (r_i, m_i)
                # core_reshaped[:, :, :] is (r_i, n_i*r', m_i)
                # Want to contract: result[k, r, m] * core[r, p, m] -> output[k, p]
                # This is: einsum('rm,rpm->p', result[k], core_reshaped)

                outputs = []
                for k in range(bOI):
                    # Contract (r_i, m_i) with (r_i, n_i*r', m_i)
                    out_k = torch.einsum('ra,rba->b', result[k], core_reshaped)  # (n_i * r')
                    outputs.append(out_k)
                result = torch.stack(outputs, dim=0)  # (bOI, n_i * r')

                # Reshape: (batch, O, I, n_i * r')
                result = result.reshape(batch_size, out_flat, in_rest_flat, n_i * r_ip1)

                # Split n_i * r': (batch, O, I, n_i, r')
                result = result.reshape(batch_size, out_flat, in_rest_flat, n_i, r_ip1)

                # Rearrange to (batch, r', O, n_i, I)
                result = result.permute(0, 4, 1, 3, 2)

                # Unflatten O and I
                new_out_dims = list(out_so_far) + [n_i]
                new_in_dims = list(in_remaining[1:])
                result = result.reshape(batch_size, r_ip1, *new_out_dims, *new_in_dims)

            # Final: (batch, r_final=1, *output_shape)
            result = result.squeeze(1)

        # Reshape output to (batch_total, out_features)
        out_features = math.prod(output_shape)
        result = result.reshape(batch_total, out_features)

        # Apply learnable scaling: per-dimension first, then global scalar
        # This preserves direction while correcting magnitude per dimension
        result = result * self.per_dim_scale * self.alpha

        # Apply inverse output permutation to restore original output channel order
        if self.output_inv_perm is not None:
            result = result[:, self.output_inv_perm]

        # Add bias if present (bias is already in permuted order, so add after unpermuting)
        if self.bias is not None:
            result = result + self.bias

        # Reshape back to original batch dimensions: (..., out_features)
        result = result.reshape(*batch_dims, out_features)

        return result

    def dense_forward(self, x):
        """
        Forward pass using the dense weight matrix reconstruction.
        This is the original implementation kept for testing purposes.
        Less efficient than forward() but useful for verification.
        """
        # Use to_matrix() which handles scaling and inverse permutation
        W = self.to_matrix()
        result = F.linear(x, W, self.bias)
        return result

    @staticmethod
    def _left_orthogonalize_core(core_i: torch.Tensor, core_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Left-orthogonalize core i and absorb the R matrix into the next core.

        Args:
            core_i: Current core with shape (r_left, n, m, r_right)
            core_next: Next core with shape (r_right, n_next, m_next, r_next)

        Returns:
            (orthogonalized core_i, modified core_next)
        """
        r_left, n, m, r_right = core_i.shape
        r_right_check, n_next, m_next, r_next = core_next.shape
        assert r_right == r_right_check, "Bond dimensions must match"

        # Reshape to matrix: (r_left * n * m, r_right)
        core_matrix = core_i.reshape(r_left * n * m, r_right)

        # QR decomposition
        Q, R = torch.linalg.qr(core_matrix)
        new_r_right = Q.shape[1]

        # Reshape Q back to core format
        new_core_i = Q.reshape(r_left, n, m, new_r_right)

        # Absorb R into next core by contracting left bond
        # core_next: (r_right, n_next, m_next, r_next)
        # R: (new_r_right, r_right)
        # Result: (new_r_right, n_next, m_next, r_next)
        core_next_matrix = core_next.reshape(r_right, n_next * m_next * r_next)
        new_core_next_matrix = R @ core_next_matrix
        new_core_next = new_core_next_matrix.reshape(new_r_right, n_next, m_next, r_next)

        return new_core_i, new_core_next

    @staticmethod
    def _right_orthogonalize_core(core_prev: torch.Tensor, core_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Right-orthogonalize core i and absorb the R matrix into the previous core.

        Args:
            core_prev: Previous core with shape (r_prev, n_prev, m_prev, r_left)
            core_i: Current core with shape (r_left, n, m, r_right)

        Returns:
            (modified core_prev, orthogonalized core_i)
        """
        r_prev, n_prev, m_prev, r_left = core_prev.shape
        r_left_check, n, m, r_right = core_i.shape
        assert r_left == r_left_check, "Bond dimensions must match"

        # Reshape to matrix: (r_left, n * m * r_right)
        core_matrix = core_i.reshape(r_left, n * m * r_right)

        # QR decomposition on transposed matrix
        Q, R = torch.linalg.qr(core_matrix.T)
        new_r_left = Q.shape[1]

        # Q.T is the orthogonalized core
        new_core_i = Q.T.reshape(new_r_left, n, m, r_right)

        # Absorb R.T into previous core by contracting right bond
        # core_prev: (r_prev, n_prev, m_prev, r_left)
        # R.T: (r_left, new_r_left)
        # Result: (r_prev, n_prev, m_prev, new_r_left)
        core_prev_matrix = core_prev.reshape(r_prev * n_prev * m_prev, r_left)
        new_core_prev_matrix = core_prev_matrix @ R.T
        new_core_prev = new_core_prev_matrix.reshape(r_prev, n_prev, m_prev, new_r_left)

        return new_core_prev, new_core_i

    def truncate_ranks(
        self,
        rank_reduction_factor: float | None = None,
        energy_threshold: float = 0.99,
        input_cov_sqrt: torch.Tensor | None = None
    ) -> "TensorizedLinear":
        """
        Truncate TT-matrix ranks using either percentage-based or energy-preserving truncation
        with proper orthogonality centering and optional data-aware truncation.

        For each bond between cores, we:
        1. Move orthogonality center to that bond using QR decompositions
        2. Reshape cores to expose the bond dimension
        3. Perform data-aware SVD if input covariance provided (V-SVD), else standard SVD
        4. Keep top singular values based on rank_reduction_factor OR energy_threshold
        5. Reconstruct cores with reduced rank

        Args:
            rank_reduction_factor: If provided, fraction to reduce ranks by (0.2 = remove 20% of rank).
                                   If None, use energy_threshold instead.
            energy_threshold: Fraction of energy to preserve (default 0.99 = 99%).
                             Only used if rank_reduction_factor is None.
            input_cov_sqrt: Optional square root of input covariance matrix for data-aware
                           truncation (V-SVD). Shape: (in_features, in_features). If provided,
                           SVD is performed on the weighted matrix W·√Σ_X to preserve directions
                           that process actual data.

        Returns:
            New TensorizedLinear with reduced ranks
        """
        orig_device = self.factors[0].device
        # Upconvert to float64 for high-precision truncation to avoid quantization loss
        factors = [f.detach().to(torch.float64) for f in self.factors]
        num_cores = len(factors)

        # Convert input_cov_sqrt to float64 if provided
        if input_cov_sqrt is not None:
            input_cov_sqrt = input_cov_sqrt.to(torch.float64)

        # Compute Frobenius norm of original matrix before truncation
        W_original = tl.tt_matrix.tt_matrix_to_matrix(factors)
        original_norm = torch.linalg.norm(W_original, ord='fro')

        # Process each bond (interface between consecutive cores)
        for k in range(num_cores - 1):
            # Step 1: Move orthogonality center to bond k
            # Left-orthogonalize all cores before k (sweep left to right)
            for i in range(k):
                factors[i], factors[i + 1] = self._left_orthogonalize_core(
                    factors[i], factors[i + 1]
                )

            # Right-orthogonalize all cores after k+1 (sweep right to left)
            for i in range(num_cores - 1, k + 1, -1):
                factors[i - 1], factors[i] = self._right_orthogonalize_core(
                    factors[i - 1], factors[i]
                )
            # Current core k has shape: (r_{k-1}, n_k, m_k, r_k)
            # Next core k+1 has shape: (r_k, n_{k+1}, m_{k+1}, r_{k+1})
            # Bond dimension is r_k

            left_core = factors[k]
            right_core = factors[k + 1]

            # Reshape left core to matrix: (r_{k-1} * n_k * m_k, r_k)
            r_left, n_k, m_k, r_bond = left_core.shape
            left_matrix = left_core.reshape(r_left * n_k * m_k, r_bond)

            # Reshape right core to matrix: (r_k, n_{k+1} * m_{k+1} * r_right)
            r_bond_check, n_kp1, m_kp1, r_right = right_core.shape
            assert r_bond == r_bond_check, "Bond dimensions must match"
            right_dims = n_kp1 * m_kp1 * r_right
            right_matrix = right_core.reshape(r_bond, right_dims)

            # Combine cores across bond: (left_dims, bond) @ (bond, right_dims)
            combined = left_matrix @ right_matrix

            # Hessian-weighted truncation (Optimal Brain Surgeon approach):
            # Instead of truncating by magnitude alone, consider sensitivity
            # Use Fisher Information Matrix (input covariance) as Hessian proxy

            # Perform SVD on unweighted matrix first to get base singular values
            U, S, Vh = torch.linalg.svd(combined, full_matrices=False)

            # Compute sensitivity-weighted importance scores
            if input_cov_sqrt is not None and right_dims == input_cov_sqrt.shape[0]:
                # Hessian-weighted: weight each singular value by its activation sensitivity
                # Project right singular vectors through input covariance to get sensitivity
                # Sensitivity ≈ how much this direction is "pushed" by real data

                # For each singular component, compute activation variance in that direction
                # Vh has shape (current_rank, right_dims)
                # input_cov_sqrt has shape (right_dims, right_dims) = (in_features, in_features)

                # Compute importance = σᵢ² · Sensitivityᵢ
                # Sensitivity = ||√C · vᵢ||² where vᵢ is right singular vector
                sensitivity_vectors = Vh @ input_cov_sqrt  # (current_rank, in_features)
                sensitivities = (sensitivity_vectors ** 2).sum(dim=1)  # (current_rank,)

                # Normalize sensitivities to make threshold interpretable
                sensitivities = sensitivities / (sensitivities.mean() + 1e-10)

                # Importance = σᵢ² · Sensitivityᵢ
                # This preserves small σᵢ if they have high sensitivity (critical for model)
                importance_scores = (S ** 2) * sensitivities
            else:
                # Standard truncation: importance = σᵢ² (raw energy only)
                importance_scores = S ** 2

            # Determine new rank and which components to keep
            current_rank = r_bond
            if rank_reduction_factor is not None:
                # Percentage-based truncation: keep top (1 - rank_reduction_factor) by magnitude
                new_rank = max(1, int(current_rank * (1.0 - rank_reduction_factor)))
                if new_rank >= current_rank:
                    new_rank = current_rank - 1
                # Keep components by magnitude order (default SVD ordering)
                keep_indices = torch.arange(new_rank, device=S.device)
            else:
                # Importance-preserving truncation (Hessian-weighted)
                # Keep enough components to preserve importance_threshold
                # Importance = σᵢ² · Sensitivityᵢ (combines magnitude and activation sensitivity)
                total_importance = importance_scores.sum()
                target_importance = energy_threshold * total_importance

                # Sort by importance (not just magnitude)
                # This is the key: we might keep a small σᵢ if it has high sensitivity
                importance_sorted_indices = torch.argsort(importance_scores, descending=True)
                cumulative_importance = torch.cumsum(importance_scores[importance_sorted_indices], dim=0)

                # Find cutoff - keep components that meet importance threshold
                meets_threshold = (cumulative_importance >= target_importance).nonzero(as_tuple=True)[0]
                if len(meets_threshold) > 0:
                    new_rank = meets_threshold[0].item() + 1
                else:
                    new_rank = current_rank

                new_rank = max(1, min(new_rank, current_rank))

                # Limit maximum rank reduction per truncation step to preserve more parameters
                # Never reduce by more than 5% in a single step
                max_rank_reduction = int(current_rank * 0.05)
                min_allowed_rank = current_rank - max_rank_reduction
                new_rank = max(new_rank, min_allowed_rank)

                # Keep most important components (may not be largest singular values!)
                keep_indices = importance_sorted_indices[:new_rank]
                # Sort kept indices to maintain proper SVD structure for reconstruction
                keep_indices = torch.sort(keep_indices)[0]

            # Truncate to selected components
            U_truncated = U[:, keep_indices]
            S_truncated = S[keep_indices]
            Vh_truncated = Vh[keep_indices, :]

            # Distribute singular values symmetrically (square root trick)
            # This keeps dynamic range balanced and prevents numerical instability
            S_sqrt = torch.sqrt(S_truncated)

            # Reconstruct left and right matrices with reduced bond dimension
            # U_new = U * sqrt(S), V_new = sqrt(S) * V
            left_matrix_new = U_truncated @ torch.diag(S_sqrt)  # (left_dims, new_rank)
            right_matrix_new = torch.diag(S_sqrt) @ Vh_truncated  # (new_rank, right_dims)

            # Reshape back to core format
            factors[k] = left_matrix_new.reshape(r_left, n_k, m_k, new_rank)
            factors[k + 1] = right_matrix_new.reshape(new_rank, n_kp1, m_kp1, r_right)

        # Apply global scaling correction to preserve Frobenius norm
        # This restores signal strength without changing direction (cosine similarity)
        W_truncated = tl.tt_matrix.tt_matrix_to_matrix(factors)
        truncated_norm = torch.linalg.norm(W_truncated, ord='fro')

        # Scaling factor: alpha = ||W_original||_F / ||W_truncated||_F
        scale_factor = original_norm / (truncated_norm + 1e-10)

        # Apply scaling to first factor (simpler and equivalent to scaling all)
        factors[0] = factors[0] * scale_factor

        # Create new TensorizedLinear with truncated factors
        new_tensorized = TensorizedLinear(
            shape=self.shape,
            rank=None,  # rank is now implicit in factors
            factors=factors,
            bias=self.bias.clone() if self.bias is not None else None,
            alpha=self.alpha.data.clone(),
            per_dim_scale=self.per_dim_scale.data.clone(),
            input_perm=self.input_perm.clone() if self.input_perm is not None else None,
            input_inv_perm=self.input_inv_perm.clone() if self.input_inv_perm is not None else None,
            dtype=self.dtype,
        )

        return new_tensorized.to(orig_device)

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])

    def to_dense(self) -> nn.Linear:
        """
        Convert this TensorizedLinear to a dense nn.Linear layer.

        Returns:
            nn.Linear with reconstructed weights (including learned scaling parameters)
        """
        with torch.no_grad():
            # to_matrix() returns (out_features, in_features) with scaling applied
            weight_matrix = self.to_matrix()

            # Get dimensions
            out_features, in_features = weight_matrix.shape
            has_bias = self.bias is not None

            # Create dense Linear layer
            dense_layer = nn.Linear(
                in_features, out_features, bias=has_bias, dtype=weight_matrix.dtype
            )

            # Copy reconstructed weights (already includes alpha and per_dim_scale)
            dense_layer.weight.data.copy_(weight_matrix)

            # Copy bias if present
            if has_bias:
                dense_layer.bias.data.copy_(self.bias)

        return dense_layer


def get_nearest_power_of_2(n: int):
    if n <= 0:
        return 1  # Handle edge cases

    # Calculate log base 2
    lg = math.log2(n)

    # Get the two closest powers
    p_lower = 2 ** int(math.floor(lg))
    p_upper = 2 ** int(math.ceil(lg))

    # Compare distances
    if (n - p_lower) < (p_upper - n):
        return p_lower
    return p_upper
