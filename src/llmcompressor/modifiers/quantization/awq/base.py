import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
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
    scale: torch.Tensor
    zero_point: Optional[torch.Tensor] = None

@dataclass
class AWQMapping:
    name: str
    layer: Module

class AWQModifier(Modifier):
    w_bit: int = 4
    group_size: int = 128
    zero_point: bool = True
    mappings: List[List[str]] = DEFAULT_AWQ_MAPPINGS
    ignore: Optional[List[str]] = None
    num_calibration_steps: Optional[int] = None
    calibration_function: Optional[Callable] = None
    
    resolved_mappings_: Optional[List] = None
    scales_: Optional[Dict] = None
    activation_stats_: Optional[Dict] = None

    def on_initialize(self, state: State, **kwargs) -> bool:
        self.ignore = [] if not self.ignore else self.ignore
        self.resolved_mappings_ = self._resolve_mappings(state.model)
        self.scales_ = {}
        self.activation_stats_ = {}

        calibration_dataloader = state.data.calib
        self._calibrate(state.model, calibration_dataloader)
        self._apply_awq(state.model)

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
        logger.info(f"Running AWQ calibration with {len(calibration_dataloader)} samples...")
        if not calibration_dataloader:
            raise ValueError("Calibration data loader not set")

        def hook_fn(module, input, output):
            module_name = self._get_module_name(module)
            if module_name not in self.activation_stats_:
                self.activation_stats_[module_name] = []
            # Capture the output (activation)
            if isinstance(output, tuple):
                output = output[0]
            self.activation_stats_[module_name].append(output.detach())

        hooks = []
        for mapping in self.resolved_mappings_:
            hooks.append(mapping.layer.register_forward_hook(hook_fn))

        run_calibration_forward(
            model,
            calibration_dataloader,
            self.num_calibration_steps,
            self.calibration_function,
        )

        for hook in hooks:
            hook.remove()


        for module_name in self.activation_stats_:
            activations = torch.cat(self.activation_stats_[module_name], dim=0)
            self.activation_stats_[module_name] = activations.abs().mean(dim=0)

    def _get_module_name(self, module):
        for mapping in self.resolved_mappings_:
            if mapping.layer == module:
                return mapping.name
        return None

    @torch.no_grad()
    def _apply_awq(self, model: Module):
        logger.info("Applying AWQ scaling...")
        for mapping in self.resolved_mappings_:
            layer = mapping.layer
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data
                activation_stats = self.activation_stats_[mapping.name]
                
                scales = self._compute_scales(weight, activation_stats)

                weight_scaled = weight / scales.unsqueeze(1)

                weight_q = torch.round(weight_scaled * (2**self.w_bit - 1))
                weight_q = torch.clamp(weight_q, 0, 2**self.w_bit - 1)

                weight_dq = weight_q / (2**self.w_bit - 1)

                weight_awq = weight_dq * scales.unsqueeze(1)

                layer.weight.data.copy_(weight_awq)

                self.scales_[mapping.name] = AWQScale(scale=scales)

    def _compute_scales(self, weight: torch.Tensor, activation_stats: torch.Tensor) -> torch.Tensor:
        weight_abs = weight.abs()

        def _compute_error(s):
            q = torch.round(weight_abs / s.unsqueeze(1) * (2**self.w_bit - 1))
            q = torch.clamp(q, 0, 2**self.w_bit - 1)
            q = q / (2**self.w_bit - 1) * s.unsqueeze(1)
            error = torch.sum(torch.abs((weight_abs - q) * activation_stats.unsqueeze(0)), dim=1)
            return error

        def _compute_scale(w):
            scale_max = w.max().item()
            scale_min = scale_max / (2**self.w_bit - 1)

            best_scale = scale_min
            best_error = float('inf')

            for i in range(20):
                scale = scale_min * (scale_max / scale_min) ** (i / 100)
                error = _compute_error(torch.tensor([scale], device=w.device))
                if error.item() < best_error:
                    best_error = error.item()
                    best_scale = scale

            return best_scale

        if self.group_size > 0:
            weight_groups = weight_abs.view(-1, self.group_size)
            activation_groups = activation_stats.view(-1, self.group_size)
            scales = torch.tensor([_compute_scale(weight_groups[i]) for i in range(weight_groups.size(0))], 
                                  device=weight.device)
            return scales.view(weight.size(0), -1).max(dim=1)[0]
        else:
            return torch.tensor([_compute_scale(weight_abs[i]) for i in range(weight.size(0))], 
                                device=weight.device)

    def on_finalize(self, state: State, **kwargs) -> bool:
        if self.scales_ is not None:
            self.scales_.clear()
        if self.resolved_mappings_ is not None:
            self.resolved_mappings_.clear()
        if self.activation_stats_ is not None:
            self.activation_stats_.clear()
        return True