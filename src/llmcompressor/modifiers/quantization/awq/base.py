from typing import List, Optional, Union

from llmcompressor.core import Event, State
from llmcompressor.modifiers import Modifier
import torch

__all__ = ["AWQModifier"]


class AWQModifier(Modifier):
    """
    Implements AWQ: Activation-aware Weight Quantization for LLM Compression
    and Acceleration https://arxiv.org/abs/2306.00978 as a Modifier in the
    LLMCompressor framework.
    """

    ignore: Optional[List[str]] = None
    preserve_ratio: float = 0.01
    scheme: str = "W4A16"
    targets: Union[str, List[str], None] = None

    def on_initialize_structure(self, state: State, **kwargs):
        pass  # nothing needed for this modifier

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run AWQ on the given state

        :param state: state to run AWQ on
        :return: True on a successful run, False otherwise
        """
        if self.end and self.end != -1:
            raise ValueError(
                f"{self.__class__.__name__} can only be applied during one-shot. "
                f" Expected end to be None or -1, got {self.end}"
            )
        if self.start and self.start != -1:
            raise ValueError(
                f"{self.__class__.__name__} can only be applied during one-shot. "
                f"Expected start to be None or -1, got {self.end}"
            )

        self.ignore = [] if not self.ignore else self.ignore

        raise NotImplementedError("Implement AWQ")

    def on_start(self, state: State, event: Event, **kwargs):
        pass

    def on_update(self, state: State, event: Event, **kwargs):
        pass

    def on_end(self, state: State, event: Event, **kwargs):
        pass

    def on_event(self, state: State, event: Event, **kwargs):
        pass

    def on_finalize(self, state: State, **kwargs) -> bool:
        raise NotImplementedError("Implement Clean up of scales")
    
    def apply(self, model, calibration_data):
        for name, module in model.named_modules():
            if any(target in name for target in self.targets) and name not in self.ignore:
                self.quantize_module(module, calibration_data)

    def quantize_module(self, module, calibration_data):
        activations = self.collect_activations(module, calibration_data)
        scale_factors, mask = self.compute_scale_factors_and_mask(activations)
        quantized_weights = self.quantize_weights(module.weight, scale_factors, mask)
        module.weight.data = quantized_weights

    def collect_activations(self, module, calibration_data):
        activations = []
        def hook(module, input, output):
            activations.append(input[0].detach())
        
        hook_handle = module.register_forward_hook(hook)
        for data in calibration_data:
            module(data)
        hook_handle.remove()
        
        activations = torch.cat(activations, dim=0)
        return activations

    def compute_scale_factors_and_mask(self, activations):
        scale_factors = torch.max(activations, dim=0).values
        threshold = torch.quantile(scale_factors, 1 - self.preserve_ratio)
        mask = scale_factors >= threshold
        return scale_factors, mask

    def quantize_weights(self, weights, scale_factors, mask):
        scaled_weights = weights / scale_factors
        quantized_weights = torch.round(scaled_weights * (2**self.num_bits - 1)) / (2**self.num_bits - 1)
        quantized_weights[mask] = weights[mask]
        return quantized_weights * scale_factors
