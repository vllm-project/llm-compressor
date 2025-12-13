import math
from typing import TypeVar, Generic
import torch
from torch import nn
import torch.nn.functional as F
import tensorly as tl
from llmcompressor.modifiers.experimental import TensorizedLinear

tl.set_backend("pytorch")

__all__ = ["BlockTensorizedLinear"]

T = TypeVar("T", nn.Linear | TensorizedLinear)


# TODO move next to compressed_tensors..CompressedLinear
class BlockTensorizedLinear(Generic[T]):
    """
    BlockLinear is an abstraction that allows a linear mapping to be reconstructed
    from constituent blocks. For example, if one wanted to break down a weight matrix into

    W = [ W11 W12
          W21 W22]

    so that W11 can be compressed.

    Original implemtntation:
    https://github.com/tensorly/Proceedings_IEEE_companion_notebooks/blob/master/tt-compression.ipynb
    """

    def __init__(
        self,
        blocks: dict[tuple[int, int], T],
        **kwargs,
    ):
        super(BlockTensorizedLinear, self).__init__(**kwargs)

        self.blocks = blocks
        self.module_dict = nn.ModuleDict(blocks)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: str | float | int | tuple[int] = 0.5,
        num_cores: int = 2,
        block_size: tuple[int, int] | None = None,
    ) -> "BlockTensorizedLinear":
        """
        Build BlockTensorizedLinear from an input torch.nn.Linear layer

        linear: original linear layer
        rank: determines the number of parameters
            if float, sets the total number of parameters to be
                linear.weight.numel() * rank
            if "same", same number of parameters as original
                (completely reconstructs the linear mapping)
        block_size: the size of each block. linear layer must have
            shape such that
                linear.shape[i] % block_size.shape[i] == 0 for any i
        """
        output_shape = cls.get_shape(linear.out_features, num_cores)
        input_shape = cls.get_shape(linear.in_features, num_cores)
        # NOTE: Order is based on what is shown in reference notebook
        # It is probably this because the linear weight matrix has
        #   a shape of (out_features, in_features)
        shape = output_shape + input_shape
        return cls(
            shape,
            rank,
            tl.decomposition.tensor_train_matrix(
                tl.reshape(linear.weight.data, shape),
                rank=rank,
            ),
            linear.bias,
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
        assert math.prod(shape) == num_features, "Something wrong with num_features"
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
