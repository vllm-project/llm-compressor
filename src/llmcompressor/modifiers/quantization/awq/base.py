from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from loguru import logger
from torch.nn import Module

from llmcompressor.core import Event, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.utils.pytorch_helpers import run_calibration_forward
from llmcompressor.utils.fsdp.helpers import get_fsdp_parent
from llmcompressor.utils.pytorch.module import get_layers, get_matching_layer

DEFAULT_AWQ_MAPPINGS = [
    ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj", "re:.*o_proj"],
    ["re:.*gate_proj", "re:.*up_proj", "re:.*down_proj"],
]

__all__ = ["AWQScale", "AWQMapping", "AWQModifier"]

@dataclass
class AWQScale:
    """
    Dataclass for storing the channel-wise scaling factors for a layer.

    :param scale: scaling factor for each output channel
    """
    scale: torch.Tensor

@dataclass
class AWQMapping:
    """
    Dataclass for storing the mapping between layers to be quantized using AWQ

    :param name: name of the layer
    :param layer: PyTorch module storing the layer
    """
    name: str
    layer: Module

class AWQModifier(Modifier):
    """
    Implements the AWQ (Adaptive Weight Quantization) algorithm from https://arxiv.org/abs/2306.00978.
    This modifier performs channel-wise scaling of weights to minimize quantization error.

    example recipe:
    ```yaml
    AWQModifier:
      w_bit: 4
      group_size: 128
      zero_point: True
      mappings: [
        ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj", "re:.*o_proj"],
        ["re:.*gate_proj", "re:.*up_proj", "re:.*down_proj"]
      ]
      ignore: []
      num_calibration_steps: 100
    ```

    :param w_bit: number of bits to use for weight quantization
    :param group_size: size of groups for groupwise quantization
    :param zero_point: whether to use zero point in quantization
    :param mappings: list of layers to apply AWQ
    :param ignore: list of layers to ignore, even if they match a regex in mappings
    :param num_calibration_steps: number of samples to use for calibration, or None to use the whole dataset
    :param calibration_function: optional function to use for the forward pass, or None to use the default tensor_module_forward
    """

    w_bit: int = 4
    group_size: int = 128
    zero_point: bool = True
    mappings: List[List[str]] = DEFAULT_AWQ_MAPPINGS
    ignore: Optional[List[str]] = None
    num_calibration_steps: Optional[int] = None
    calibration_function: Optional[Callable] = None

    resolved_mappings_: Optional[List] = None
    scales_: Optional[Dict] = None

    def on_initialize_structure(self, state: State, **kwargs):
        pass

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.end and self.end != -1:
            raise ValueError(f"{self.__class__.__name__} can only be applied during one-shot. Expected end to be None or -1, got {self.end}")
        if self.start and self.start != -1:
            raise ValueError(f"{self.__class__.__name__} can only be applied during one-shot. Expected start to be None or -1, got {self.end}")

        self.ignore = [] if not self.ignore else self.ignore
        self.resolved_mappings_ = self._resolve_mappings(state.model)
        self.scales_ = {}

        calibration_dataloader = state.data.calib
        self._calibrate(state.model, calibration_dataloader)
        self._apply_awq(state.model)

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        pass

    def on_update(self, state: State, event: Event, **kwargs):
        pass

    def on_end(self, state: State, event: Event, **kwargs):
        pass

    def on_event(self, state: State, event: Event, **kwargs):
        pass

    def on_finalize(self, state: State, **kwargs) -> bool:
        if self.scales_ is not None:
            self.scales_.clear()
        if self.resolved_mappings_ is not None:
            self.resolved_mappings_.clear()

        return True

    def _resolve_mappings(self, model: Module) -> List:
        resolved_mappings = []
        for mapping in self.mappings:
            for layer_pattern in mapping:
                layers = get_layers(layer_pattern, model)
                for layer_name, layer in layers.items():
                    if layer_name not in self.ignore:
                        resolved_mappings.append(AWQMapping(layer_name, layer))
        return resolved_mappings

    @torch.no_grad()
    def _calibrate(self, model: Module, calibration_dataloader: List):
        class_name = self.__class__.__name__.replace("PyTorch", "")
        logger.info(f"Running {class_name} calibration with {len(calibration_dataloader)} samples...")
        if not calibration_dataloader:
            raise ValueError("Calibration data loader not set, must populate the calib_data field of CompressionSession to run the AWQ modifier")

        run_calibration_forward(
            model,
            calibration_dataloader,
            self.num_calibration_steps,
            self.calibration_function,
        )

    @torch.no_grad()
    def _apply_awq(self, model: Module):
        logger.info("Applying AWQ scaling...")
        for mapping in self.resolved_mappings_:
            layer = mapping.layer
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data
                in_features = weight.shape[1]
                out_features = weight.shape[0]

                # Compute scales
                if self.group_size > 0:
                    num_groups = in_features // self.group_size
                    weight_groups = weight.view(out_features, num_groups, self.group_size)
                    scales = weight_groups.abs().max(dim=2)[0]
                else:
                    scales = weight.abs().max(dim=1)[0]

                # Apply scaling
                weight_scaled = weight / scales.unsqueeze(1)

                # Quantize
                if self.zero_point:
                    weight_min = weight_scaled.min(dim=1, keepdim=True)[0]
                    weight_max = weight_scaled.max(dim=1, keepdim=True)[0]
                    weight_scaled = (weight_scaled - weight_min) / (weight_max - weight_min)
                
                weight_q = torch.round(weight_scaled * (2**self.w_bit - 1))
                weight_q = torch.clamp(weight_q, 0, 2**self.w_bit - 1)

                # Dequantize
                if self.zero_point:
                    weight_dq = weight_q / (2**self.w_bit - 1) * (weight_max - weight_min) + weight_min
                else:
                    weight_dq = weight_q / (2**self.w_bit - 1)

                # Apply scaling back
                weight_awq = weight_dq * scales.unsqueeze(1)

                # Update layer weight
                layer.weight.data.copy_(weight_awq)

                # Store scaling factors
                self.scales_[mapping.name] = AWQScale(scale=scales)

    def _calculate_awq_scales(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Calculate AWQ scaling factors for the given weight tensor.

        :param weight: weight tensor to calculate scaling factors for
        :return: channel-wise scaling factors
        """
        in_features = weight.shape[1]
        if self.group_size > 0:
            num_groups = in_features // self.group_size
            weight_groups = weight.view(-1, num_groups, self.group_size)
            scales = weight_groups.abs().max(dim=2)[0]
        else:
            scales = weight.abs().max(dim=1)[0]
        return scales