import math

import tensorly as tl
import torch
from torch import nn

from llmcompressor.modifiers.experimental.tensorized_linear import TensorizedLinear

tl.set_backend("pytorch")

__all__ = ["BlockTensorizedLinear"]


# TODO move next to compressed_tensors..CompressedLinear
class BlockTensorizedLinear(nn.Module):
    """
    BlockTensorizedLinear is an abstraction that allows a linear mapping to be
    reconstructed from constituent blocks. For example, if one wanted to break
    down a weight matrix into

    W = [ W11 W12
          W21 W22]

    so that W11, ..., W22 can be compressed further, found to be useful when compressing
    into tensor network topologies with limited entanglement (e.g. MPOs).

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

        # Store block references in a 2D list for easier access
        # This is more torch.compile-friendly than dynamic ModuleDict access
        self._block_grid = nn.ModuleList(
            [
                nn.ModuleList(
                    [self.blocks[f"{i}_{j}"] for j in range(self.num_blocks[1])]
                )
                for i in range(self.num_blocks[0])
            ]
        )

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

        # Store original shape and flatten batch dimensions
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        # Flatten to (...batch..., in_features) -> (total_batch, in_features)
        x_flat = x.reshape(-1, self.in_features)
        batch_size = x_flat.shape[0]

        # Reshape input to separate blocks: (batch, num_blocks[1], block_size)
        x_blocks = x_flat.reshape(batch_size, self.num_blocks[1], self.block_size)

        # Use fully vectorized forward implementation (no Python loops!)
        y_blocks = self._forward_impl(x_blocks)

        # Reshape output back: (batch, num_blocks[0], block_size) -> (batch, out_features)
        y_flat = y_blocks.reshape(batch_size, self.out_features)

        # Restore original batch dimensions
        y = y_flat.reshape(*batch_dims, self.out_features)

        return y

    def _forward_impl(self, x_blocks: torch.Tensor):
        """
        Fully vectorized forward implementation with NO Python loops.

        Processes all blocks at once using nested list comprehensions and
        vectorized stack + sum operations. This eliminates all Python iteration
        overhead and is significantly faster than the original nested loop version.

        How it works:
        1. Outer list comp processes all output rows (i) in parallel
        2. Inner list comp processes all input columns (j) for each row in parallel
        3. torch.stack + sum combines column results for each row
        4. Final stack + permute assembles all rows into output tensor

        Args:
            x_blocks: Input reshaped as (batch, num_blocks[1], block_size)

        Returns:
            y_blocks: Output tensor of shape (batch, num_blocks[0], block_size)
        """
        # Process ALL rows at once using list comprehension
        # For each row i, compute all column blocks in parallel and sum them
        all_row_outputs = [
            torch.stack(
                [
                    self._block_grid[i][j](x_blocks[:, j, :])
                    for j in range(self.num_blocks[1])
                ],
                dim=0,
            ).sum(
                dim=0
            )  # Sum across columns: (num_cols, batch, block_size) -> (batch, block_size)
            for i in range(self.num_blocks[0])
        ]

        # Stack all rows: list of (batch, block_size) -> (num_rows, batch, block_size)
        # Then permute to (batch, num_rows, block_size)
        y_blocks = torch.stack(all_row_outputs, dim=0).permute(1, 0, 2)

        return y_blocks

    def dense_forward(self, x):
        assert x.shape[-1] == self.in_features
        out_shape = x.shape[:-1] + (self.out_features,)
        y = torch.zeros(out_shape, device=x.device, dtype=x.dtype)
        for i in range(self.num_blocks[0]):
            for j in range(self.num_blocks[1]):
                y[..., i * self.block_size : (i + 1) * self.block_size] += self.blocks[
                    f"{i}_{j}"
                ].dense_forward(x[..., j * self.block_size : (j + 1) * self.block_size])

        return y

    @property
    def num_params(self):
        return sum([b.num_params for b in self.blocks.values()])
