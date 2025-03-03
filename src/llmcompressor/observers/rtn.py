from typing import Any, Optional, Tuple

import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.utils import calculate_qparams
from compressed_tensors.utils import deprecated

from llmcompressor.observers.base import Observer
from llmcompressor.pytorch.utils import pseudo_quantize_tensor

__all__ = ["RoundToNearestObserver"]


@Observer.register("rtn")
class RoundToNearestObserver(Observer):
    """
    Implements a quantization observer that calculates scale and zero point based on the
    minimum and maximum values of the tensor being observed. If averaging_constant is
    specified, then the scales are updated using a moving average
    """

    def calculate_qparams(
        self,
        observed: torch.Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
        tensor_id: Optional[Any] = None,
    ) -> Tuple[torch.FloatTensor, torch.IntTensor]:
        """
        Updates the observed min and max using a moving average smoothed by the
        averaging_constant. Set the averaging_constant to 1.0 to disable averaging.

        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned scale and zero point will be shaped (1,) along the
            reduced dimensions
        :param tensor_id: Optional id if different ranges of observed tensors are
            passed, useful for sharding tensors by group_size
        :return: tuple of scale and zero point derived from the observed tensor
        """

        _, scales, zp = pseudo_quantize_tensor(
            observed,
            symmetric=self.quantization_args.symmetric,
            bit_width=self.quantization_args.num_bits,
            group_size=-1, #self.quantization_args.group_size,
        )
        return (scales, zp)

    def get_qparams_along_dim(
        self, observed: torch.Tensor, dim: int, tensor_id: Optional[Any] = None
    ):
        """
        Calculate quantization parameters along the specified dimension
        """
        reduce_dims = tuple(idx for idx in range(observed.ndim) if idx != dim)
        return self.calculate_qparams(
            observed, reduce_dims=reduce_dims, tensor_id=tensor_id
        )
