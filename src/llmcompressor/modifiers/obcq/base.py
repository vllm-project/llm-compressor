import contextlib
from typing import Dict, List, Optional, Tuple, Union

import torch
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    update_offload_parameter,
)
from loguru import logger
from pydantic import Field, PrivateAttr

from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.obcq.sgpt_mixin import SparsityModifierMixin
from llmcompressor.modifiers.obcq.sgpt_sparsify import (
    accumulate_hessian,
    make_empty_hessian,
    sparsify_weight,
)
from llmcompressor.utils.metric_logging import CompressionLogger

__all__ = ["SparseGPTModifier"]


class SparseGPTModifier(SparsityModifierMixin, Modifier):
    """
    Modifier for applying the one-shot SparseGPT algorithm to a model

    Lifecycle:
        - on_initialize
            - initialize_compression()
                - compressible_layers()
                - LayerCompressor.pre_compress()
            - apply_compression()
                - run_calibration_forward()
                - LayerCompressor.compress()
                - LayerCompressor.post_compress()
                - LayerCompressor.revert_layer_wrappers()

    | Sample yaml:
    |   test_stage:
    |       obcq_modifiers:
    |           SparseGPTModifier:
    |               sparsity: 0.5
    |               mask_structure: "2:4"
    |               sequential_update: True
    |               dampening_frac: 0.001
    |               block_size: 128

    :param sparsity: Sparsity to compress model to
    :param sparsity_profile: Can be set to 'owl' to use Outlier Weighed
        Layerwise Sparsity (OWL), more information can be found
        in the paper https://arxiv.org/pdf/2310.05175
    :param owl_m: Number of outliers to use for OWL
    :param owl_lmbda: Lambda value to use for OWL
    :param mask_structure: String to define the structure of the mask to apply.
        Must be of the form N:M where N, M are integers that define a custom block
        shape. Defaults to 0:0 which represents an unstructured mask.
    :param sequential_update: Whether or not to update weights sequentially by layer,
        True saves on GPU memory
    :param targets: list of layer names to compress during OBCQ, or '__ALL__'
        to compress every layer in the model
    :param block_size: Used to determine number of columns to compress in one pass
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param preserve_sparsity_mask: Whether or not to preserve the sparsity mask
        during when applying sparsegpt, this becomes useful when starting from a
        previously pruned model, defaults to False.
    """

    # modifier arguments
    sparsity: Optional[Union[float, List[float]]] = None
    mask_structure: str = "0:0"
    owl_m: Optional[int] = None
    owl_lmbda: Optional[float] = None  # misspelling?
    sparsity_profile: Optional[str] = None  # deprecated
    block_size: int = 128
    dampening_frac: Optional[float] = 0.01
    preserve_sparsity_mask: bool = False  # deprecate?
    offload_hessians: bool = False

    # data pipeline arguments
    sequential_update: Optional[bool] = False  # deprecated
    module_targets: Union[str, List[str]] = ["Linear"]
    targets: Union[str, List[str], None] = None  # deprecated, clones sequential_targets
    sequential_targets: Union[str, List[str], None] = None
    ignore: List[str] = Field(default_factory=list)

    # private variables
    _prune_n: Optional[int] = PrivateAttr(default=None)
    _prune_m: Optional[int] = PrivateAttr(default=None)
    _hessians: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _module_sparsities: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)

    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ):
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

    def on_sequential_batch_end(self):
        """
        Sparsify modules
        TODO: implement with event callback
        """
        for module in list(self._num_samples.keys()):
            name = self._module_names[module]
            sparsity = self._module_sparsities[module]
            num_samples = self._num_samples[module]

            logger.info(f"Sparsifying {name} using {num_samples} samples")
            with (
                torch.no_grad(),
                align_module_device(module),
                CompressionLogger(module) as comp_logger,
            ):
                loss, sparsified_weight = sparsify_weight(
                    module=module,
                    hessians_dict=self._hessians,
                    sparsity=sparsity,
                    prune_n=self._prune_n,
                    prune_m=self._prune_m,
                    block_size=self.block_size,
                    dampening_frac=self.dampening_frac,
                    preserve_sparsity_mask=self.preserve_sparsity_mask,  # TODO: should we deprecate this? GPTQ just checks against a sparsity threshold
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
