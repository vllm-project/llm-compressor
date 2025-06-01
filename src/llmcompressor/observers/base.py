from math import ceil
from typing import Any, Iterable, Optional, Tuple, Union

import torch
from compressed_tensors.quantization.quant_args import (
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationStrategy,
)
from compressed_tensors.quantization.utils import is_fp4
from compressed_tensors.registry.registry import RegistryMixin
from compressed_tensors.utils import safe_permute
from loguru import logger
from torch import FloatTensor, IntTensor, Tensor
from torch.nn import Module

__all__ = ["Observer"]


class Observer(Module, RegistryMixin):
    """
    Base Observer class to be subclassed for specific implementation.
    Subclasses should override `calculate_qparams` to return a scale, zero_point
    pair
    """

    def __init__(
        self,
        quantization_args: QuantizationArgs,
    ):
        self.quantization_args: QuantizationArgs = quantization_args
        super().__init__()
        self._scale = None
        self._zero_point = None
        self._num_observed_tokens = None

    @torch.no_grad()
    def forward(
        self,
        observed: Tensor,
        g_idx: Optional[Tensor] = None,
        global_scale: Optional[Tensor] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        maps directly to get_qparams
        :param observed: optional observed tensor from which to calculate
            quantization parameters
        :param g_idx: optional mapping from column index to group index
        :param global_scale: optional scale to further scale local quantization scales
        :return: tuple of scale and zero point based on last observed value
        """
        self.record_observed_tokens(observed)
        return self.get_qparams(
            observed=observed, g_idx=g_idx, global_scale=global_scale
        )

    def calculate_qparams(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
        tensor_id: Optional[Any] = None,
        global_scale: Optional[Tensor] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """Calculate quantization parameters for the observed tensor.

        Args:
            observed: Tensor to calculate quantization parameters for
            reduce_dims: Optional tuple of dimensions to reduce along.
                Returned scale and zero point will be shaped (1,) along the
                reduced dimensions
            tensor_id: Optional identifier for the tensor or tensor part being
                quantized
            global_scale: Optional scale to further scale local quantization scales

        Returns:
            Tuple of scale and zero point derived from the observed tensor
        """
        if self.quantization_args.strategy == QuantizationStrategy.BLOCK:
            # Parse block structure - but ONLY if this is a top-level call
            if tensor_id is None:
                # This is the top-level call - handle the block structure
                block_structure = self.quantization_args.block_structure
                if block_structure is None:
                    raise ValueError(
                        "block_structure must be specified for block-wise quantization"
                    )

                try:
                    block_rows, block_cols = map(int, block_structure.split("x"))
                    if block_rows <= 0 or block_cols <= 0:
                        raise ValueError(
                            f"Block dimensions must be positive integers, "
                            f"got {block_rows}x{block_cols}"
                        )
                except (ValueError, AttributeError):
                    raise ValueError(
                        f"Invalid block_structure: {block_structure}, "
                        "expected format 'AxB' (e.g. '128x128')"
                    )

                rows, columns = observed.shape
                if observed.ndim != 2:
                    raise ValueError(
                        f"Block-wise quantization expects 2D tensors, "
                        f"got tensor with {observed.ndim} dimensions"
                    )

                num_row_blocks = ceil(rows / block_rows)
                num_col_blocks = ceil(columns / block_cols)

                # Check if dimensions are multiples of block size
                if (
                    num_row_blocks * block_rows != rows
                    or num_col_blocks * block_cols != columns
                ):
                    logger.bind(log_once=True).warning(
                        f"Tensor dimensions ({rows}x{columns}) are not divisible by "
                        f"block_structure ({block_structure}). Padding will be applied."
                    )

                # Create tensors to hold scales and zero points
                scale_tensor = torch.zeros_like(observed)
                zero_point_tensor = torch.zeros_like(observed, dtype=torch.int32)

                # Process each block
                for row_block in range(num_row_blocks):
                    row_start = row_block * block_rows
                    row_end = min(row_start + block_rows, rows)

                    for col_block in range(num_col_blocks):
                        col_start = col_block * block_cols
                        col_end = min(col_start + block_cols, columns)

                        # Get block data
                        block_data = observed[row_start:row_end, col_start:col_end]

                        # Calculate min/max for this block
                        block_min = block_data.min()
                        block_max = block_data.max()

                        # Calculate scale and zero point for this block
                        if block_max == block_min:
                            block_scale = torch.tensor(1.0, device=observed.device)
                            block_zero_point = torch.tensor(
                                0, dtype=torch.int32, device=observed.device
                            )
                        else:
                            # For int8, qmin=-128, qmax=127
                            qmin, qmax = -128, 127  # Default for INT8
                            block_scale = (block_max - block_min) / (qmax - qmin)
                            block_zero_point = torch.round(
                                qmin - block_min / block_scale
                            ).to(torch.int32)

                        # Extract scalar values if needed
                        if hasattr(block_scale, "item"):
                            block_scale = block_scale.item()
                        if hasattr(block_zero_point, "item"):
                            block_zero_point = block_zero_point.item()

                        # Fill the corresponding region in the tensors
                        scale_tensor[row_start:row_end, col_start:col_end].fill_(
                            block_scale
                        )
                        zero_point_tensor[row_start:row_end, col_start:col_end].fill_(
                            block_zero_point
                        )

                # Store the full tensors
                self._scale = scale_tensor
                self._zero_point = zero_point_tensor.to(
                    dtype=(
                        FP8_E4M3_DATA.dtype
                        if is_fp4(quantization_args=self.quantization_args)
                        else self.quantization_args.pytorch_dtype()
                    )
                )

                return self._scale, self._zero_point
            else:
                # This is a recursive call for a specific block
                min_val = observed.min()
                max_val = observed.max()

                # For int8, qmin=-128, qmax=127
                qmin, qmax = -128, 127  # Default for INT8

                if max_val == min_val:
                    scale = torch.tensor(1.0, device=observed.device)
                    zero_point = torch.tensor(
                        0, dtype=torch.int32, device=observed.device
                    )
                else:
                    scale = (max_val - min_val) / (qmax - qmin)
                    zero_point = torch.round(qmin - min_val / scale).to(torch.int32)

                return scale, zero_point
        else:
            # For non-block quantization, use global min/max
            min_val = observed.min()
            max_val = observed.max()

            # For int8, qmin=-128, qmax=127
            qmin, qmax = -128, 127  # Default for INT8

            if max_val == min_val:
                scale = torch.tensor(1.0, device=min_val.device)
                zero_point = torch.tensor(0, dtype=torch.int32, device=min_val.device)
            else:
                scale = (max_val - min_val) / (qmax - qmin)
                zero_point = torch.round(qmin - min_val / scale).to(torch.int32)

            return scale, zero_point

    def post_calculate_qparams(self) -> None:
        """
        Run any logic specific to its observers after running calculate_qparams
        """

    def get_qparams(
        self,
        observed: Optional[Tensor] = None,
        g_idx: Optional[Tensor] = None,
        global_scale: Optional[Tensor] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """Get quantization parameters for the observed tensor.

        Args:
            observed: Optional tensor to calculate quantization parameters from
            g_idx: Optional mapping from column index to group index
            global_scale: Optional scale to further scale local quantization scales

        Returns:
            Tuple of scale and zero point based on last observed value
        """
        if observed is not None:
            group_size = self.quantization_args.group_size

            if self.quantization_args.strategy == QuantizationStrategy.TENSOR:
                # re-calculate scale and zero point, update the stored value
                self._scale, self._zero_point = self.calculate_qparams(observed)

            elif self.quantization_args.strategy in (
                QuantizationStrategy.TENSOR_GROUP,
                QuantizationStrategy.GROUP,
            ):
                rows = observed.shape[0]
                columns = observed.shape[1]
                num_groups = int(ceil(columns / group_size))
                if num_groups * group_size != columns:
                    logger.bind(log_once=True).warning(
                        "Attempting to quantize a module weight whose columns "
                        f"({columns}) are not divisible by group_size ({group_size}). "
                        "This scheme is not supported by vLLM, please consider "
                        "adjusting the group_size for modules with this number of "
                        "columns",
                    )

                self._scale = torch.empty(
                    (rows, num_groups), dtype=observed.dtype, device=observed.device
                )
                if is_fp4(quantization_args=self.quantization_args):
                    zp_dtype = FP8_E4M3_DATA.dtype
                else:
                    zp_dtype = self.quantization_args.pytorch_dtype()

                self._zero_point = torch.empty(
                    (rows, num_groups), dtype=zp_dtype, device=observed.device
                )

                # support column-order (default) quantization as well as other orderings
                # such as activation ordering. Below checks if g_idx has initialized
                is_column_order = g_idx is None or -1 in g_idx
                if is_column_order:
                    group_sizes = torch.full((num_groups,), group_size, dtype=torch.int)
                else:
                    group_indices, group_sizes = torch.unique(g_idx, return_counts=True)
                    group_sizes = group_sizes[torch.argsort(group_indices)]

                    perm = torch.argsort(g_idx)
                    observed = safe_permute(observed, perm, dim=1)

                # TODO: experiment with vectorizing for loop for performance
                end = 0
                for group_index, group_count in enumerate(group_sizes):
                    start = end
                    end = start + group_count
                    scale, zero_point = self.get_qparams_along_dim(
                        observed[:, start:end],
                        0,
                        tensor_id=group_index,
                        global_scale=global_scale,
                    )

                    self._scale[:, group_index] = scale.squeeze(1)
                    self._zero_point[:, group_index] = zero_point.squeeze(1)

            elif self.quantization_args.strategy == QuantizationStrategy.CHANNEL:
                # assume observed is transposed, because its the output, hence use dim 0
                self._scale, self._zero_point = self.get_qparams_along_dim(observed, 0)

            elif self.quantization_args.strategy == QuantizationStrategy.TOKEN:
                # use dim 1, assume the obsersed.shape = [batch, token, hidden]
                # should be batch, token
                self._scale, self._zero_point = self.get_qparams_along_dim(
                    observed,
                    dim={0, 1},
                )

            elif self.quantization_args.strategy == QuantizationStrategy.BLOCK:
                self._scale, self._zero_point = self.calculate_qparams(
                    observed, tensor_id=None, global_scale=global_scale
                )
            else:
                raise ValueError(
                    f"Unsupported quantization strategy: "
                    f"{self.quantization_args.strategy}"
                )

        return self._scale, self._zero_point

    def get_qparams_along_dim(
        self,
        observed,
        dim: Union[int, Iterable[int]],
        tensor_id: Optional[Any] = None,
        global_scale: Optional[Tensor] = None,
    ):
        if isinstance(dim, int):
            dim = [dim]
        dim = set(dim)

        reduce_dims = tuple(idx for idx in range(observed.ndim) if idx not in dim)
        return self.calculate_qparams(
            observed,
            reduce_dims=reduce_dims,
            tensor_id=tensor_id,
            global_scale=global_scale,
        )

    def record_observed_tokens(self, batch_tensor: Tensor):
        """
        Counts the number of tokens observed during the
        forward passes. The count is aggregated in the
        _num_observed_tokens attribute of the class.

        Note: The batch_tensor is expected to have two dimensions
            (batch_size * sequence_length, num_features). This is the
            general shape expected by the forward pass of the expert
            layers in a MOE model. If the input tensor does not have
            two dimensions, the _num_observed_tokens attribute will be set
            to None.
        """
        if not isinstance(batch_tensor, Tensor):
            raise ValueError(f"Expected value to be a tensor, got {type(batch_tensor)}")

        if batch_tensor.ndim != 2:
            logger.debug(
                "The input tensor is expected to have two dimensions "
                "(batch_size * sequence_length, num_features). "
                f"The input tensor has {batch_tensor.ndim} dimensions."
            )
            return

        if self._num_observed_tokens is None:
            # initialize the count
            self._num_observed_tokens = 0

        # batch_tensor (batch_size * sequence_length, num_features)
        # observed_tokens (batch_size * sequence_length)
        observed_tokens, _ = batch_tensor.shape
        self._num_observed_tokens += observed_tokens

    def reset(self):
        """
        Reset the state of the observer
        """
        self._num_observed_tokens = None
        self._scale = None
        self._zero_point = None
