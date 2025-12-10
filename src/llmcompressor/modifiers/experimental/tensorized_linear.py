import torch
from torch import nn
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
        **kwargs
    ):
        super(TensorizedLinear, self).__init__(**kwargs)

        self.rank = rank
        self.shape = shape

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
        cls, linear: nn.Linear, rank: int | tuple[int] | None
    ) -> "TensorizedLinear":
        output_shape = (linear.out_features // 2, linear.out_features // 2)
        input_shape = (linear.in_features // 2, linear.in_features // 2)
        shape = output_shape + input_shape
        tt_matrix = tl.decomposition.tensor_train_matrix(
            tl.reshape(linear.weight.data, shape),
            rank=rank,
        )
        return cls(shape, rank, tt_matrix)

    def to_matrix(self) -> torch.Tensor:
        """
        Return tensorized weights expanded into a single weight matrix
        """
        return self.tt_matrix.to_matrix()

    def forward(self, x):
        # TODO use einsum string instead of rebuilding matrix
        # form full weight matrix
        W = tl.tt_matrix.tt_matrix_to_matrix(self.factors)

        # perform regular matrix multiplication
        return torch.matmul(x, W)

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])
