from math import ceil
from typing import Any, Iterable, Optional, Tuple, Union

import torch
from compressed_tensors import InternalModule
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

__all__ = ["Observer"]


class Observer(InternalModule, RegistryMixin):
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
        should_calculate_gparam: bool = False,
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
        if should_calculate_gparam:
            return self.get_gparam(observed=observed)
        return self.get_qparams(
            observed=observed,
            g_idx=g_idx,
            global_scale=global_scale,
        )

    def calculate_qparams(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned scale and zero point will be shaped (1,) along the
            reduced dimensions
        :return: tuple of scale and zero point derived from the observed tensor
        """
        raise NotImplementedError(f"{self.__class__} must implement calculate_qparams")

    def calculate_gparam(
        self,
        observed: Tensor,
    ) -> torch.Tensor:
        """
        :param observed: observed tensor to calculate quantization parameters for
        :return: global scale derived from the observed tensor
        """
        raise NotImplementedError(f"{self.__class__} must implement calculate_gparam")

    def post_calculate_qparams(self) -> None:
        """
        Run any logic specific to its observers after running calculate_qparams
        """

    def get_gparam(self, observed: Tensor):
        """
        Function to derive a global scale parameter
        :param observed: observed tensor to calculate global parameters
            from
        :return: derived global scale
        """
        if self.quantization_args.strategy == QuantizationStrategy.TENSOR_GROUP:
            return self.calculate_gparam(observed)
        raise NotImplementedError(
            "global parameter generation is only supported for TENSOR_GROUP"
        )

    def get_qparams(
        self,
        observed: Optional[Tensor] = None,
        g_idx: Optional[Tensor] = None,
        global_scale: Optional[Tensor] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        Convenience function to wrap overwritten calculate_qparams
        adds support to make observed tensor optional and support for tracking latest
        calculated scale and zero point

        :param observed: optional observed tensor to calculate quantization parameters
            from
        :param g_idx: optional mapping from column index to group index
        :param global_scale: optional scale to further scale local quantization scales
        :return: tuple of scale and zero point based on last observed value
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
                # TODO (#1475) add support for block-wise quantization
                raise NotImplementedError(
                    "Block-wise quantization is not yet supported, "
                    "consider group-wise quantization instead. More info at "
                    "https://github.com/vllm-project/llm-compressor/issues/1475"
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
