import math
import torch
from torch import nn
import torch.nn.functional as F
import tensorly as tl
from llmcompressor.modifiers.experimental import TensorizedLinear

tl.set_backend("pytorch")

__all__ = ["BlockTensorizedLinear"]


# TODO move next to compressed_tensors..CompressedLinear
class BlockTensorizedLinear(nn.Module):
    """
    BlockTensorizedLinear is an abstraction that allows a linear mapping to be reconstructed
    from constituent blocks. For example, if one wanted to break down a weight matrix into

    W = [ W11 W12
          W21 W22]

    so that W11, ..., W22 can be compressed further, found to be useful when compressing into
    tensor network topologies with limited entanglement (e.g. MPOs).

    Original implemtntation:
    https://github.com/tensorly/Proceedings_IEEE_companion_notebooks/blob/master/tt-compression.ipynb
    """

    def __init__(
        self,
        # TODO: make Generic instead of hard-coded to TensorizedLinear
        # Using strings as keys here to be compatible with torch.nn.ModuleDict
        blocks: dict[str, TensorizedLinear],
        block_size: int,
        num_blocks: tuple[int, int],
        in_features: int,
        out_features: int,
        **kwargs,
    ):
        super(BlockTensorizedLinear, self).__init__(**kwargs)

        self.blocks = nn.ModuleDict(blocks)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.in_features = in_features
        self.out_features = out_features

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int,
        # Args to pass into TensorizedLinear.from_linear
        rank: str | float | int | tuple[int] = 0.5,
        num_cores: int = 2,
    ) -> "BlockTensorizedLinear":
        """
        Build BlockTensorizedLinear from an input torch.nn.Linear layer

        linear: original linear layer
        block_size: the size of each block. linear layer must have
            shape such that
                linear.shape[i] % block_size == 0 for any i
            For now, hard-coding num_rows == num_columns
            # TODO allow for block_size: tuple[int, int]
        rank: same as TensorizedLinear.from_linear
        num_cores: same as TensorizedLinear.from_linear
        """
        assert (
            linear.in_features % block_size == 0
        ), "invalid block size for in_features"
        assert (
            linear.out_features % block_size == 0
        ), "invalid block size for out_features"

        num_rows = linear.out_features // block_size
        num_cols = linear.in_features // block_size

        blocks: dict[tuple[int, int], TensorizedLinear] = {}
        for i in range(num_rows):
            for j in range(num_cols):
                blocks[f"{i}_{j}"] = TensorizedLinear.from_weight_and_bias(
                    weight=linear.weight[
                        i * block_size : (i + 1) * block_size,
                        j * block_size : (j + 1) * block_size,
                    ],
                    # only need to add bias on first block in row
                    bias=(
                        linear.bias[i * block_size : (i + 1) * block_size]
                        if (linear.bias is not None and j == 0)
                        else None
                    ),
                    rank=rank,
                    num_cores=num_cores,
                )
        return cls(
            blocks,
            block_size,
            (num_rows, num_cols),
            linear.in_features,
            linear.out_features,
        )

    def to_matrix(self) -> torch.Tensor:
        """
        Return tensorized weights expanded into a single weight matrix
        """
        matrix_chunks = []
        for i in range(self.num_blocks[0]):
            row_chunks = [
                self.blocks[f"{i}_{j}"].to_matrix() for j in range(self.num_blocks[1])
            ]
            matrix_chunks.append(torch.cat(row_chunks, dim=1))
        return torch.cat(matrix_chunks)

    def forward(self, x):
        assert x.shape[-1] == self.in_features
        out_shape = x.shape[:-1] + (self.out_features,)
        y = torch.zeros(out_shape)
        for i in range(self.num_blocks[0]):
            for j in range(self.num_blocks[1]):
                y[..., i * self.block_size : (i + 1) * self.block_size] += self.blocks[
                    f"{i}_{j}"
                ](x[..., j * self.block_size : (j + 1) * self.block_size])

        return y

    @property
    def num_params(self):
        return sum([b.num_params() for b in self.blocks.values()])


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
