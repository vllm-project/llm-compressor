import math
import torch
from torch import nn
import torch.nn.functional as F
import tensorly as tl
from tensorly.tt_matrix import TTMatrix

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
        tt_matrix: TTMatrix | None,
        bias: torch.Tensor | None = None,
        **kwargs,
    ):
        super(TensorizedLinear, self).__init__(**kwargs)

        self.rank = rank
        self.shape = shape
        self.bias = bias

        # if not provided, initialize tt_matrix and weights
        if tt_matrix is None:
            tt_matrix = tl.random.random_tt_matrix(shape, rank=rank)
            for f in tt_matrix.factors:
                f.data.uniform_(-0.1, 0.1)

        self.tt_matrix = tt_matrix
        # Add and register the factors
        self.factors = nn.ParameterList(
            [nn.Parameter(f, requires_grad=True) for f in tt_matrix.factors]
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
        return cls.from_weight_and_bias(linear.weight, linear.bias, rank, num_cores)

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
        return cls(
            shape,
            rank,
            tl.decomposition.tensor_train_matrix(
                tl.reshape(weight, shape),
                rank=rank,
            ),
            bias,
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
                dim = get_nearest_power_of_2(remainder ** (1 / (num_cores - i)))
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
        return self.tt_matrix.to_matrix()

    def forward(self, x):
        # TODO use einsum string instead of rebuilding matrix
        # form full weight matrix
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
