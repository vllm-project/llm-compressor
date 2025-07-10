from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from compressed_tensors.utils import align_module_device
from loguru import logger
from pydantic import ConfigDict, Field
from torch.nn import Module

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.smoothquant.utils import (
    get_layer_mappings_from_architecture,
    handle_mapping_resolution_errors,
)
from llmcompressor.utils.fsdp.helpers import get_fsdp_parent
from llmcompressor.utils.pytorch.module import (
    get_layers,
    get_matching_layer,
    match_targets,
)

MINIMUM_SMOOTHING_SCALE = 1e-5


__all__ = ["SmoothQuantScale", "SmoothQuantMapping", "SmoothQuantModifier"]


@dataclass
class SmoothQuantScale:
    """
    Dataclass for storing the channel-wise minimum and maximum values for a layer. This
    is updated each forward pass during calibration

    :param min_channel_vals: minimum output value seen so far, per channel
    :param max_channel_vals: maximum output value seen so far, per channel
    """

    min_channel_vals: torch.Tensor
    max_channel_vals: torch.Tensor


@dataclass
class SmoothQuantMapping:
    """
    Dataclass for storing the mapping between an activation layer and the following
    weights that must be balanced during smoothing

    :param smooth_name: name of the activation layer
    :param smooth_layer: PyTorch module storing the activation layer
    :param balance_layers: list of PyTorch modules that smooth_layer feeds into, must be
    balanced to offset the smoothing of smooth_layer
    """

    smooth_name: str
    smooth_layer: Module
    balance_layers: List[Module]


class SmoothQuantModifier(Modifier):
    """
     Implements the SmoothQuant algorithm from https://arxiv.org/abs/2211.10438. This
     modifier performs a channel-wise smoothing of outliers in activations, making them
     easier to quantize by reducing the dynamic range. The smoothing is offset by
     applying the inverse operation to the next layer of weights, making the weights
     slightly more difficult to quantize.

     Because this modifier manipulates the weights of the model, it can only be used in
     in one-shot and not during training. Activation ranges are determined by running a
     small set of calibration data through the model.

    example recipe:
     ```yaml
     SmoothQuantModifier:
       smoothing_strength: 0.5
       mappings: [
         [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*self_attn_layer_norm"],
         [["re:.*fc1"], "re:.*final_layer_norm"]
       ]
       ignore: ["model.decoder.final_layer_norm"]
     ```

     :param smoothing_strength: alpha, intensity of smoothing to perform (0-1 range)
     :param mappings: list activation layers to smooth, and which layers to
        scale the output such that activations are smoothed.
        Each entry of the mapping list should be a list itself, in which the first
        entry is a list of layers who share the same input activation (the one to be
        to smoothed) and the second entry is the layer whose output is scaled to
        achieve the smoothing. If regex is used, it matches layers with the largest
        overlap in module name.  If not supplied the argument will be inferred from the
        model architecture.
     :param ignore: list of layers to ignore, even if they match a regex in mappings.
        It should match the name of layers whose outputs are scaled to achieve
        smoothing (the second entry of the mappings list).
     :param num_calibration_steps: number of samples to use for calibration, or None to
     use the whole dataset
    :param calibration_function: optional function to use for the forward pass, or None
    to use the default tensor_module_forward
    """

    smoothing_strength: float = 0.5
    mappings: Optional[List[Union[Tuple, List]]] = None
    ignore: Optional[List[str]] = None
    num_calibration_steps: Optional[int] = None
    calibration_function: Optional[Callable] = None

    resolved_mappings_: Optional[List[SmoothQuantMapping]] = Field(
        default=None, repr=False
    )
    scales_: Optional[Dict] = Field(default=None, repr=False)

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run SmoothQuant on the given state

        :param state: state to run SmoothQuant on
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

        if not hasattr(state, "data") or state.data.calib is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a calibration dataset to be "
                "provided"
            )
        self.ignore = [] if not self.ignore else self.ignore
        self.mappings = self._infer_mappings_from_model(state.model)
        self.resolved_mappings_ = self._resolve_mappings(state.model)
        self.scales_ = {}

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True
        self._setup_scale_hooks()

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self._apply_smoothing(state.model)

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            self._apply_smoothing(state.model)

            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        self.ended_ = True
        self.remove_hooks()  # remove hooks

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        Clean up by clearing the scale and mapping data
        """
        if not self.ended_:
            self.on_end(state, None)

        if len(self.scales_) > 0:
            raise ValueError(f"Failed to compress {len(self.scales_)} modules")

        if self.scales_ is not None:
            self.scales_.clear()
        if self.resolved_mappings_ is not None:
            self.resolved_mappings_.clear()

        return True

    def _infer_mappings_from_model(
        self,
        model: Module,
    ) -> List[Tuple]:
        if self.mappings is not None:
            return self.mappings

        logger.info("No SmoothQuantModifier.mappings provided, inferring from model...")
        return get_layer_mappings_from_architecture(
            architecture=model.__class__.__name__
        )

    @handle_mapping_resolution_errors
    def _resolve_mappings(self, model: Module) -> List[SmoothQuantMapping]:
        """
        Transforms the list of activations to smooth and their corresponding weights
        into SmoothQuantMapping objects, resolving regular expressions.

        For each activation in the mapping list, we find the corresponding weight to
        balance by searching for the longest substring. For instance, if our balance
        weight is ".*re:.*q_proj" and the activation is "re:.*self_attn_layer_norm" we
        would match model.layer.0.p_proj to model.layer.0.self_attn_layer_norm and
        repeat for model.layer.1 and so on
        """
        resolved_mappings = []
        for to_balance, to_smooth in self.mappings:
            to_smooth_layers = get_layers(to_smooth, model)
            for layer_name, smooth_layer in to_smooth_layers.items():
                if not match_targets(layer_name, self.ignore)[0]:
                    balance_layers = []
                    for balance_suffix in to_balance:
                        # find the submodule that matches the activation layer
                        _, balance_layer = get_matching_layer(
                            balance_suffix, layer_name, model
                        )
                        if balance_layer:
                            balance_layers.append(balance_layer)
                    # each mapping can contain multiple layers to balance, but only
                    # one layer to smooth
                    mapping = SmoothQuantMapping(
                        layer_name, smooth_layer, balance_layers
                    )
                    resolved_mappings.append(mapping)
        return resolved_mappings

    def _setup_scale_hooks(self):
        """
        Attach a forward hook to each activation we want to smooth. This allows us to
        calculate the dynamic range during calibration
        """

        def create_hook_fn(layer_name):
            def hook_fn(module, inp, out):
                # update the per-channel min/max output values seen during calibration
                if isinstance(out, tuple):
                    out = out[0]

                hidden_dim = out.shape[-1]
                out = out.view(-1, hidden_dim)
                latest_mins = torch.min(out, dim=0)[0]
                latest_maxes = torch.max(out, dim=0)[0]

                if layer_name in self.scales_:
                    self.scales_[layer_name].min_channel_vals = torch.minimum(
                        self.scales_[layer_name].min_channel_vals, latest_mins
                    )
                    self.scales_[layer_name].max_channel_vals = torch.maximum(
                        self.scales_[layer_name].max_channel_vals, latest_maxes
                    )
                else:
                    self.scales_[layer_name] = SmoothQuantScale(
                        min_channel_vals=latest_mins, max_channel_vals=latest_maxes
                    )

            return hook_fn

        for mapping in self.resolved_mappings_:
            name = mapping.smooth_name
            layer = mapping.smooth_layer
            self.register_hook(layer, create_hook_fn(name), "forward")

    @torch.no_grad()
    def _apply_smoothing(self, model: Module):
        """
        After calibration, apply smoothing to the activations and push the transform
        into the following weights by applying the inverse to each balance weight.

        Y = (Xdiag(scales)^(-1) * diag(scales)W) where W is the to_balance weights and
        X is the to_smooth weights

        This modifies the weights of the model in-place.
        """
        for mapping in self.resolved_mappings_:
            if mapping.smooth_name not in self.scales_:
                continue
            logger.info(f"Smoothing with {mapping.smooth_name}")

            activation_scales = (  # get dynamic range for each activation channel
                self.scales_[mapping.smooth_name].max_channel_vals
                - self.scales_[mapping.smooth_name].min_channel_vals
            )
            smooth_layer = mapping.smooth_layer
            balance_layers = mapping.balance_layers

            scales = self._calculate_smoothing_scales(balance_layers, activation_scales)
            scales = torch.maximum(
                scales, torch.Tensor([MINIMUM_SMOOTHING_SCALE]).to(scales.device)
            )

            @torch.no_grad()
            def smooth(module):
                with align_module_device(module):
                    if module in balance_layers:
                        module.weight.mul_(scales.view(1, -1))
                    elif module == smooth_layer:
                        if module.weight.ndim == 1:
                            module.weight.div_(scales)
                        else:
                            module.weight.div_(scales.view(-1, 1))
                        if hasattr(module, "bias") and module.bias is not None:
                            module.bias.div_(scales)

            parent = get_fsdp_parent(mapping.smooth_name, model)
            if parent is not None:
                parent.apply(smooth)
            else:
                # if we're not running with FSDP we can apply smoothing directly
                for layer in balance_layers:
                    smooth(layer)
                smooth(smooth_layer)

            # clear calibration data
            del self.scales_[mapping.smooth_name]

    def _calculate_smoothing_scales(
        self, balance_layers: List[Module], activation_scales: torch.Tensor
    ) -> List[float]:
        """
        Calculate how much smoothing to apply to each channel based on the dynamic
        range of the activation and the following weights

        :param balance_layers: layers to offset activation smoothing to
        :param activation_scales: channel-wise dynamic range of activations to smooth
        :return: channel-wise scales to use for smoothing activations
        """
        # get the channel-wise dynamic range for each layer to be balanced
        weight_scales = []
        for layer in balance_layers:
            with align_module_device(layer):
                scale = layer.weight.abs().max(dim=0, keepdim=True)[0]
                weight_scales.append(scale)

        weight_scales = 2.0 * torch.cat(weight_scales, dim=0).max(dim=0)[0]

        # calculate the amount of smoothing to apply
        # s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
        # where j is the input channel, alpha is smoothing strength
        scales = activation_scales.pow(self.smoothing_strength) / weight_scales.pow(
            1 - self.smoothing_strength
        )
        scales = torch.where(weight_scales > 0.0, scales, activation_scales)

        return scales

    model_config = ConfigDict(arbitrary_types_allowed=True)
