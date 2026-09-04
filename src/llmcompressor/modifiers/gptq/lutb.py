from typing import ClassVar

import torch
from compressed_tensors.quantization.quant_args import ActivationOrdering

from llmcompressor.core import Event, State
from llmcompressor.modifiers.gptq.base import GPTQModifier
from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight_lut_b

__all__ = ["GPTQLutBModifier"]


class GPTQLutBModifier(GPTQModifier):
    """
    GPTQ variant for LUT-B (lookup-table / codebook) weight quantization.

    Behaves exactly like :class:`GPTQModifier` -- it calibrates a hessian from
    input activations and quantizes weights column-by-column with hessian-based
    error propagation -- but fits a non-uniform E4M3 codebook per weight tile
    instead of a uniform scale/zero-point grid. See :class:`GPTQModifier` for the
    full description of parameters and lifecycle.

    Differences from :class:`GPTQModifier`:

    - Weights are quantized via ``quantize_weight_lut_b`` (codebook fitting)
      rather than ``quantize_weight``.
    - No weight/activation observers are run: the codebook is fit inside the
      quantization routine, so there are no scale/zero-point qparams to observe.
    - The only quantization parameter produced is ``weight`` (the LUT-B
      compressor recovers an identical codebook at save time), so nothing else
      needs to be broadcast/stored in the distributed path.
    - Activation ordering is unsupported: LUT-B uses 2D block tiling
      (``[block_n, block_k]``) and actorder would scatter a tile's columns.
    """

    # LUT-B uses 2D block tiling; activation ordering would scatter a tile's
    # columns, so it is unsupported and defaults to off (base default is static).
    actorder: ActivationOrdering | None = None

    # LUT-B produces a non-uniform codebook, not a scale/zero-point grid. The
    # final weight takes on E4M3 center values, so the LUT-B compressor recovers
    # an identical codebook at save time and only the weight is broadcast/stored.
    _q_param_names: ClassVar[list[str]] = ["weight"]

    def on_sequential_epoch_end(
        self, state: State, event: Event, modules: list[torch.nn.Module], **kwargs
    ):
        # LUT-B is weight-only and codebook-based: no weight or activation
        # observers to run (the codebook is fit inside quantize_weight_lut_b).
        self.compress_modules()

    def _quantize_weight(self, module, quant_args, hessian):
        return quantize_weight_lut_b(
            module=module,
            quant_args=quant_args,
            hessian=hessian,
            blocksize=self.block_size,
            percdamp=self.dampening_frac,
        )
