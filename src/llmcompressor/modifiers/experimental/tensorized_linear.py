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
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super(TensorizedLinear, self).__init__(**kwargs)

        self.rank = rank
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

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: str | float | int | tuple[int] = 0.5,
        num_cores: int = 2,
    ) -> "TensorizedLinear":
        """
        Build TensorizedLinear from an input torch.nn.Linear layer


        linear: original linear layer's weight matrix
        rank: determines the number of parameters
            if float, sets the total number of parameters to be
                linear.weight.numel() * rank
            if "same", same number of parameters as original
                (completely reconstructs the linear mapping)
        """
        return cls.from_weight_and_bias(
            linear.weight.detach(),
            linear.bias.detach() if linear.bias is not None else None,
            rank,
            num_cores,
        )

    @classmethod
    def from_weight_and_bias(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        rank: str | float | int | tuple[int] = 0.5,
        num_cores: int = 2,
    ) -> "TensorizedLinear":
        """
        Build TensorizedLinear from an input torch.nn.Linear layer

        linear: original linear layer's weight matrix
        rank: determines the number of parameters
            if float, sets the total number of parameters to be
                linear.weight.numel() * rank
            if "same", same number of parameters as original
                (completely reconstructs the linear mapping)
        """
        assert weight.ndim == 2, "invalid weight"
        # always create fresh
        weight = weight.clone(memory_format=torch.contiguous_format)

        if bias is not None:
            assert bias.ndim == 1, "invalid bias"
            assert weight.shape[0] == bias.shape[0], "incompatible weight/bias"
            bias = bias.clone(memory_format=torch.contiguous_format)

        output_shape = cls.get_shape(weight.shape[0], num_cores)
        input_shape = cls.get_shape(weight.shape[1], num_cores)
        # NOTE: Order is based on what is shown in reference notebook
        # It is probably this because the linear weight matrix has
        #   a shape of (out_features, in_features)
        shape = output_shape + input_shape

        # upconvert to float32 before svd, no bfloat16 implementation
        orig_dtype = weight.dtype
        weight = weight.to(torch.float32)
        tt_matrix = tl.decomposition.tensor_train_matrix(
            tl.reshape(weight, shape),
            rank=rank,
        )
        return cls(
            shape,
            rank,
            tt_matrix.factors,
            bias,
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
        Return tensorized weights expanded into a single weight matrix
        """
        return tl.tt_matrix.tt_matrix_to_matrix(self.factors)

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
        x_flat = x.reshape(-1, in_features)
        batch_total = x_flat.shape[0]

        # Get shapes from factors
        input_shape = [f.shape[2] for f in self.factors]
        output_shape = [f.shape[1] for f in self.factors]
        num_cores = len(self.factors)

        # Reshape input to expose individual dimensions: (batch, m_0, m_1, ..., m_{d-1})
        x_reshaped = x_flat.reshape(batch_total, *input_shape)

        # Build einsum string using single-character labels (torch.einsum requirement)
        # Assign characters: a-z for various indices
        # 'b' = batch
        # Next num_cores chars for input dims (m_i)
        # Next num_cores chars for output dims (n_i)
        # Next num_cores+1 chars for rank dims (r_i)

        input_chars = [chr(ord("c") + i) for i in range(num_cores)]  # c, d, e, ...
        output_chars = [chr(ord("C") + i) for i in range(num_cores)]  # C, D, E, ...
        rank_chars = [chr(ord("p") + i) for i in range(num_cores + 1)]  # p, q, r, ...

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
        result = torch.einsum(einsum_string, x_reshaped, *self.factors)

        # Reshape output to (batch_total, out_features)
        out_features = math.prod(output_shape)
        result = result.reshape(batch_total, out_features)

        # Add bias if present
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
        # Form full weight matrix
        W = tl.tt_matrix.tt_matrix_to_matrix(self.factors)
        return F.linear(x, W, self.bias)

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])


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
