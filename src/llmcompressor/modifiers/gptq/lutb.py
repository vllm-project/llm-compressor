from typing import ClassVar

import torch
from compressed_tensors.quantization.quant_args import ActivationOrdering
from compressed_tensors.offload import disable_offloading
from compressed_tensors.utils import getattr_chain, update_offload_parameter

from llmcompressor.core import Event, State
from llmcompressor.modifiers.gptq.base import GPTQModifier
from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight_lut_b
from llmcompressor.utils.metric_logging import CompressionLogger

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
    - The only GPTQ parameter produced is ``weight``. LUT-B tiles are range
      normalized with E8M0 scales during fitting; the CT compressor writes the
      final codebooks and scales from the resulting quantized weights.
    - Activation ordering is unsupported: LUT-B uses 2D block tiling
      (``[block_n, block_k]``) and actorder would scatter a tile's columns.
    """

    # LUT-B uses 2D block tiling; activation ordering would scatter a tile's
    # columns, so it is unsupported and defaults to off (base default is static).
    actorder: ActivationOrdering | None = None

    # LUT-B produces a non-uniform codebook, not a scale/zero-point grid. The
    # output weight contains scaled E4M3 center values, so no separate GPTQ
    # qparams need to be broadcast; CT emits the canonical book and E8M0 scale.
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

    def compress_module_list(self, module_list):
        """Quantize LUT-B modules one at a time without observer qparams."""
        for module in module_list:
            name = self._module_names[module]
            quant_args = getattr_chain(module, "quantization_scheme.weights")
            hessian = self._hessians.pop(module)
            num_samples = self._num_samples.pop(module).to(device=hessian.device)

            with (
                torch.no_grad(),
                disable_offloading(),
                CompressionLogger(module) as comp_logger,
            ):
                loss, q_param_dict, used_rtn_fallback = self._quantize_weight(
                    module=module,
                    quant_args=quant_args,
                    hessian=hessian / num_samples,
                )
                comp_logger.set_results(name="GPTQ", loss=loss)

            if used_rtn_fallback:
                self._rtn_fallback_module_names.append(name)
            for attr, value in q_param_dict.items():
                update_offload_parameter(module, attr, value)

            self._num_compressed_modules += 1
