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
from llmcompressor.modifiers.pruning.wanda.utils.wanda_sparsify import (
    accumulate_row_scalars,
    make_empty_row_scalars,
    sparsify_weight,
)
from llmcompressor.utils.metric_logging import CompressionLogger

__all__ = ["WandaPruningModifier"]


class WandaPruningModifier(SparsityModifierMixin, Modifier):
    """
    Modifier for applying the one-shot WANDA algorithm to a model
    from the paper: https://arxiv.org/abs/2306.11695

    :param sparsity: Sparsity to compress model to
    :param mask_structure: String to define the structure of the mask to apply.
        Must be of the form N:M where N, M are integers that define a custom block
        shape. Defaults to 0:0 which represents an unstructured mask.
    :param sequential_update: Whether or not to update weights sequentially by layer,
        True saves on GPU memory
    :param targets: list of layer names to compress during OBCQ, or '__ALL__'
        to compress every layer in the model
    """

    # sparsity arguments
    sparsity: Optional[Union[float, List[float]]] = None
    mask_structure: str = "0:0"
    owl_m: Optional[int] = None
    owl_lmbda: Optional[float] = None  # misspelling?
    sparsity_profile: Optional[str] = None  # deprecated

    # data pipeline arguments
    module_targets: Union[str, List[str]] = ["Linear"]
    targets: Union[str, List[str], None] = None  # deprecated, clones sequential_targets
    sequential_targets: Union[str, List[str], None] = None
    ignore: List[str] = Field(default_factory=list)

    # private variables
    _prune_n: Optional[int] = PrivateAttr(default=None)
    _prune_m: Optional[int] = PrivateAttr(default=None)
    _row_scalars: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(
        default_factory=dict
    )
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

        # Initialize row scalars if not present
        if module not in self._num_samples:
            device = get_execution_device(module)
            self._row_scalars[module] = make_empty_row_scalars(module, device=device)
            self._num_samples[module] = 0

        # Accumulate scalars using data
        self._row_scalars[module], self._num_samples[module] = accumulate_row_scalars(
            inp,
            module,
            self._row_scalars[module],
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
                CompressionLogger(module),
            ):
                sparsified_weight = sparsify_weight(
                    module=module,
                    row_scalars_dict=self._row_scalars,
                    sparsity=sparsity,
                    prune_n=self._prune_n,
                    prune_m=self._prune_m,
                )

            update_offload_parameter(module, "weight", sparsified_weight)

            # self._row_scalars[module] already deleted by sparsify_weight
            del self._num_samples[module]
