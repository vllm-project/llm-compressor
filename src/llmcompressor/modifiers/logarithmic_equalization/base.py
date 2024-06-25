from typing import List

import torch
from torch.nn import Module

from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

__all__ = ["LogarithmicEqualizationModifier"]


class LogarithmicEqualizationModifier(SmoothQuantModifier):
    """
     Implements the Logarithmic Equalization Algorithm from
     https://arxiv.org/abs/2308.15987.
     This modifier performs a channel-wise smoothing of outliers in activations,
     making them easier to quantize by reducing the dynamic range. The smoothing is
     offset by applying the inverse operation to the next layer of weights, making
     the weights slightly more difficult to quantize.

     Because this modifier manipulates the weights of the model, it should only be
     used in one-shot and not during training. Activation ranges are determined by
     running a small set of calibration data through the model.

     This algorithm is very similar to SmoothQuant, changing only how the smoothing
     scales are computed. This modifier inherits most functionality from the
     SmoothQuantModifier.

    example recipe:
     ```yaml
     LogarithmicEqualizationModifier:
       mappings: [
         [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*self_attn_layer_norm"],
         [["re:.*fc1"], "re:.*final_layer_norm"]
       ]
       ignore: ["model.decoder.final_layer_norm"]
     ```

    :param mappings: list activation layers to smooth, and which layers to
      scale the output such that activations are smoothed.
      Each entry of the mapping list should be a list itself, in which the first
      entry is a list of layers who share the same input activation (the one to be
      to smoothed) and the second entry is the layer whose output is scaled to
      achieve the smoothing.
      If regex is used, it matches layers with the largest overlap in module name.
    :param ignore: list of layers to ignore, even if they match a regex in mappings.
      It should match the name of layers whose outputs are scaled to achieve
      smoothing (the second entry of the mappings list).
    :param num_calibration_steps: number of samples to use for calibration, or None to
      use the whole dataset
    :param calibration_function: optional function to use for the forward pass, or None
      to use the default tensor_module_forward
    """

    def _calculate_smoothing_scales(
        self, balance_layers: List[Module], activation_scales: torch.Tensor
    ) -> List[float]:
        """
        Calculate how much smoothing to apply to each channel based on the dynamic
        range of the activations and the following weights.

        :param balance_layers: layers to offset activation smoothing to
        :param activation_scales: channel-wise dynamic range of activations to smooth
        :return: channel-wise scales to use for smoothing activations
        """
        # calculate the amount of smoothing to apply
        # s_j = max(|X_j|) / log2( 2 + max(|X_j|) )
        # where j is the input channel
        scales = activation_scales / torch.log2(2 + activation_scales)
        return scales
