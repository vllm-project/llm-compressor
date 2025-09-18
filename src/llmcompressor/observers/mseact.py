from typing import Any, Optional, Tuple

import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs, FP4_E2M1_DATA, FP8_E4M3_DATA
from compressed_tensors.quantization.utils import calculate_qparams
from torch import FloatTensor, IntTensor, Tensor

from llmcompressor.observers.base import Observer

__all__ = ["MovingAverageMSEObserverActivations"]


@Observer.register("mseact")
class MovingAverageMSEObserverActivations(Observer):
    """
    Implements a dynamic quantization observer that sets the scale and
    zero point based on a moving average of the mse-clipped min and max observed values
    """

    def __init__(
        self,
        quantization_args: QuantizationArgs,
        maxshrink: float = 0.2,
        patience: int = 5,
        averaging_constant: float = 0.01,
        grid: float = 100.0,
        norm: float = 2.4,
        **kwargs,
    ):
        super().__init__(quantization_args=quantization_args)

        self.min_val = {}
        self.max_val = {}
        self.maxshrink = maxshrink
        self.patience = patience
        self.averaging_constant = averaging_constant
        self.grid = grid
        self.norm = norm

    def _generate_dynamic_gparam(
        self,
        updated_min_val: torch.Tensor,
        updated_max_val: torch.Tensor,
        scale_data = FP8_E4M3_DATA,
        quant_data = FP4_E2M1_DATA,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Generate a dynamic global scale using MSE-optimized min/max values.
        This implementation uses the MSE-optimized values instead of the static min/max approach.
        
        :param updated_min_val: MSE-optimized minimum values
        :param updated_max_val: MSE-optimized maximum values  
        :param scale_data: FP8 data configuration for scales
        :param quant_data: FP4 data configuration for quantization
        :param dtype: output dtype for the global scale
        :return: dynamically calculated global scale
        """
        min_vals = torch.min(updated_min_val, torch.zeros_like(updated_min_val))
        max_vals = torch.max(updated_max_val, torch.zeros_like(updated_max_val))
        max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))
        max_val_pos = torch.clamp(max_val_pos, min=1e-12)
        global_scale = scale_data.max * quant_data.max / max_val_pos
        return global_scale.to(dtype).reshape([1])

    def calculate_mse_min_max(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
        global_scale: Optional[torch.Tensor] = None,
    ):
        """
        Computes the mse-clipped min and max values of the observed tensor by
        optimizing for quantization error

        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned values will be shaped (1,) along the reduced dimensions
        :param global_scale: optional scale to further scale local quantization scales
        :return: tuple of min and max values derived from the observed tensor
        """

        from compressed_tensors.quantization.lifecycle import fake_quantize

        if not reduce_dims:
            absolute_min_val, absolute_max_val = torch.aminmax(observed)
        else:
            absolute_min_val = torch.amin(observed, dim=reduce_dims, keepdims=True)
            absolute_max_val = torch.amax(observed, dim=reduce_dims, keepdims=True)

        best = torch.full_like(
            absolute_min_val, torch.finfo(absolute_min_val.dtype).max
        )
        min_val = torch.ones_like(absolute_min_val)
        max_val = torch.zeros_like(absolute_max_val)

        # Early stopping params
        no_improve_count = 0
        total_iterations = int(self.maxshrink * self.grid)

        for i in range(int(self.maxshrink * self.grid)):
            p = 1 - i / self.grid
            shrinked_min_val = p * absolute_min_val
            shrinked_max_val = p * absolute_max_val

            # Generate dynamic global scale using shrinked values
            iteration_global_scale = self._generate_dynamic_gparam(
                updated_min_val=shrinked_min_val, updated_max_val=shrinked_max_val
            )

            # Use shrinked values for BOTH global scale and local scales (consistent approach)
            candidate_scales, candidate_zero_points = calculate_qparams(
                min_vals=shrinked_min_val,  # Use shrinked values (consistent with global scale)
                max_vals=shrinked_max_val,  # Use shrinked values (consistent with global scale)
                quantization_args=self.quantization_args,
                global_scale=iteration_global_scale,
            )
            q = fake_quantize(
                observed,
                candidate_scales,
                candidate_zero_points,
                self.quantization_args,
                global_scale=iteration_global_scale,
            )

            q -= observed
            q.abs_()
            q.pow_(self.norm)
            if not reduce_dims:
                err = torch.sum(q)
            else:
                err = torch.sum(q, reduce_dims, keepdims=True)

            tmp = err < best
            if torch.any(tmp):
                best[tmp] = err[tmp]
                min_val[tmp] = shrinked_min_val[tmp]
                max_val[tmp] = shrinked_max_val[tmp]
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    break

        return min_val, max_val

    def calculate_updated_min_max(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
        tensor_id: Optional[Any] = None,
        global_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        Updates the mse-clipped min and max values of the observed tensor using
        a moving average smoothed by the averaging_constant

        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned scale and zero point will be shaped (1,) along the
            reduced dimensions
        :param tensor_id: Optional id if different ranges of observed tensors are
            passed, useful for sharding tensors by group_size
        :param global_scale: optional scale to further scale local quantization scales
        :return: updated min and max values derived from the observed value
        """
        min_val, max_val = self.calculate_mse_min_max(
            observed, reduce_dims, global_scale=global_scale
        )

        tensor_id = tensor_id or "default"

        running_min_val = self.min_val.get(tensor_id, None)
        running_max_val = self.max_val.get(tensor_id, None)

        if running_min_val is None or running_max_val is None:
            updated_min_val = min_val
            updated_max_val = max_val
        else:
            updated_min_val = running_min_val + self.averaging_constant * (
                min_val - running_min_val
            )
            updated_max_val = running_max_val + self.averaging_constant * (
                max_val - running_max_val
            )

        self.min_val[tensor_id] = updated_min_val
        self.max_val[tensor_id] = updated_max_val
        return updated_min_val, updated_max_val

    def calculate_qparams(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
        tensor_id: Optional[Any] = None,
        global_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        Updates the mse-clipped min and max values of the observed tensor using
        a moving average smoothed by the averaging_constant

        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned scale and zero point will be shaped (1,) along the
            reduced dimensions
        :param tensor_id: Optional id if different ranges of observed tensors are
            passed, useful for sharding tensors by group_size
        :param global_scale: optional scale to further scale local quantization scales
        :return: tuple of scale and zero point derived from the observed tensor
        """

        updated_min_val, updated_max_val = self.calculate_updated_min_max(
            observed=observed,
            tensor_id=tensor_id,
            reduce_dims=reduce_dims,
            global_scale=global_scale,
        )
        scale, zero_point = calculate_qparams(
            min_vals=updated_min_val,
            max_vals=updated_max_val,
            quantization_args=self.quantization_args,
            global_scale=global_scale,
        )

        return scale, zero_point

    def get_qparams_along_dim(
        self,
        observed,
        dim: int,
        tensor_id: Optional[Any] = None,
        global_scale: Optional[torch.Tensor] = None,
    ):
        reduce_dims = tuple(idx for idx in range(observed.ndim) if idx != dim)
        return self.calculate_qparams(
            observed,
            reduce_dims=reduce_dims,
            tensor_id=tensor_id,
            global_scale=global_scale,
        )

    def reset(self):
        """
        Reset the state of the observer, including min and maximum values
        """
        super().reset()
        self.min_val = {}
        self.max_val = {}

    def calculate_gparam(self, observed: torch.Tensor) -> torch.Tensor:
        """
        Generate a dynamic global scale using the observed min and max from MSE optimization.
        This method is called during calibration to calculate the global scale on-the-fly
        using MSE-optimized min/max values instead of static min/max.

        :param observed: observed tensor to calculate quantization parameters for
        :return: dynamically calculated global scale derived from MSE-optimized values
        """
                
        # Get MSE-optimized min/max values using moving average
        updated_min_val, updated_max_val = self.calculate_updated_min_max(
            observed=observed
        )
    
        # Generate dynamic global scale using MSE-optimized values
        global_scale = self._generate_dynamic_gparam(
            updated_min_val=updated_min_val,
            updated_max_val=updated_max_val,
        )
        
        return global_scale
