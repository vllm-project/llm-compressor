import contextlib
from typing import Dict, Optional, Tuple

import torch
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    update_offload_parameter,
)
from loguru import logger
from pydantic import PrivateAttr

from llmcompressor.core import State
from llmcompressor.modifiers.obcq.sgpt_base import SparsityModifierBase
from llmcompressor.modifiers.obcq.sgpt_sparsify import (
    accumulate_hessian,
    make_empty_hessian,
    sparsify_weight,
)
from llmcompressor.utils.metric_logging import CompressionLogger

__all__ = ["SparseGPTModifier"]


class SparseGPTModifier(SparsityModifierBase):
    """
    Modifier for applying the one-shot SparseGPT algorithm to a model

    | Sample yaml:
    |   test_stage:
    |       obcq_modifiers:
    |           SparseGPTModifier:
    |               sparsity: 0.5
    |               mask_structure: "2:4"
    |               dampening_frac: 0.001
    |               block_size: 128
    |               targets: ['Linear']
    |               ignore: ['re:.*lm_head']

    Lifecycle:
        - on_initialize
            - register_hook(module, calibrate_module, "forward")
        - on_sequential_batch_end
            - sparsify_weight
        - on_finalize
            - remove_hooks()

    :param sparsity: Sparsity to compress model to
    :param sparsity_profile: Can be set to 'owl' to use Outlier Weighed
        Layerwise Sparsity (OWL), more information can be found
        in the paper https://arxiv.org/pdf/2310.05175
    :param mask_structure: String to define the structure of the mask to apply.
        Must be of the form N:M where N, M are integers that define a custom block
        shape. Defaults to 0:0 which represents an unstructured mask.
    :param owl_m: Number of outliers to use for OWL
    :param owl_lmbda: Lambda value to use for OWL
    :param block_size: Used to determine number of columns to compress in one pass
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param preserve_sparsity_mask: Whether or not to preserve the sparsity mask
        during when applying sparsegpt, this becomes useful when starting from a
        previously pruned model, defaults to False.
    :param offload_hessians: Set to True for decreased memory usage but increased
        runtime.
    :param sequential_targets: list of layer names to compress during OBCQ, or '__ALL__'
        to compress every layer in the model. Alias for `targets`
    :param targets: list of layer names to compress during OBCQ, or '__ALL__'
        to compress every layer in the model. Alias for `sequential_targets`
    :param ignore: optional list of module class names or submodule names to not
        quantize even if they match a target. Defaults to empty list.
    """

    # modifier arguments
    block_size: int = 128
    dampening_frac: Optional[float] = 0.01
    preserve_sparsity_mask: bool = False
    offload_hessians: bool = False

    # private variables
    _num_samples: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)

    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ):
        """
        Calibration hook used to accumulate the hessian of the input to the module

        :param module: module being calibrated
        :param args: inputs to the module, the first element of which is the
            cannonical input
        :param _output: uncompressed module output, unused
        """
        # Assume that the first argument is the input
        inp = args[0]

        # Initialize hessian if not present
        if module not in self._num_samples:
            device = get_execution_device(module)
            self._hessians[module] = make_empty_hessian(module, device=device)
            self._num_samples[module] = 0

        # Accumulate hessian with input with optional offloading
        with self._maybe_onload_hessian(module):
            self._hessians[module], self._num_samples[module] = accumulate_hessian(
                inp,
                module,
                self._hessians[module],
                self._num_samples[module],
            )

    def compress_modules(self):
        """
        Sparsify modules which have been calibrated
        """
        for module in list(self._num_samples.keys()):
            name = self._module_names[module]
            sparsity = self._module_sparsities[module]
            num_samples = self._num_samples[module]

            logger.info(f"Sparsifying {name} using {num_samples} samples")
            with torch.no_grad(), align_module_device(module), CompressionLogger(
                module
            ) as comp_logger:
                loss, sparsified_weight = sparsify_weight(
                    module=module,
                    hessians_dict=self._hessians,
                    sparsity=sparsity,
                    prune_n=self._prune_n,
                    prune_m=self._prune_m,
                    block_size=self.block_size,
                    dampening_frac=self.dampening_frac,
                    preserve_sparsity_mask=self.preserve_sparsity_mask,
                )
                comp_logger.set_loss(loss)

            update_offload_parameter(module, "weight", sparsified_weight)

            # self._hessians[module] already deleted by sparsify_weight
            del self._num_samples[module]

    @contextlib.contextmanager
    def _maybe_onload_hessian(self, module: torch.nn.Module):
        if self.offload_hessians:
            device = get_execution_device(module)
            self._hessians[module] = self._hessians[module].to(device=device)

        yield

        if self.offload_hessians:
            if module in self._hessians:  # may have been deleted in context
                self._hessians[module] = self._hessians[module].to(device="cpu")

    def on_finalize(self, state: State, **kwargs) -> bool:
        # TODO: modify lifecycle to end on finalize
        if not self.ended_:
            self.on_end(state, None)  # remove hooks

        if len(self._num_samples) > 0:
            raise ValueError(f"Failed to compress {len(self._num_samples)} modules")

        self._hessians = dict()
        self._num_samples = dict()
        self._module_names = dict()
        self._module_sparsities = dict()

        return True
