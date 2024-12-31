from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from torch.nn import Module
from tqdm import tqdm
from collections import defaultdict

from llmcompressor.core import State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.pruning.wanda.utils.wanda_wrapper import WandaWrapper
from llmcompressor.modifiers.utils.layer_compressor import LayerCompressor
from llmcompressor.modifiers.utils.pytorch_helpers import run_calibration_forward
from llmcompressor.utils.pytorch.module import (
    get_layers,
    get_no_split_params,
    get_prunable_layers,
)

__all__ = ["WandaPruningModifier"]


class WandaPruningModifier(Modifier):
    """
    Modifier for applying the one-shot WANDA algorithm to a model
    from the paper: https://arxiv.org/abs/2306.11695

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
        - on_finalize

    :param sparsity: Sparsity to compress model to
    :param mask_structure: String to define the structure of the mask to apply.
        Must be of the form N:M where N, M are integers that define a custom block
        shape. Defaults to 0:0 which represents an unstructured mask.
    :param sequential_update: Whether or not to update weights sequentially by layer,
        True saves on GPU memory
    :param targets: list of layer names to compress during OBCQ, or '__ALL__'
        to compress every layer in the model
    """

    sparsity: Union[float, List[float]] = 0.0
    sparsity_profile: Optional[str] = None
    owl_m: Optional[int] = None
    owl_lmbda: Optional[float] = None
    mask_structure: str = "0:0"
    sequential_update: Optional[bool] = False
    sequential_targets: Union[str, List[str], None] = None
    model: Optional[Any] = None

    compressible_layers_: Optional[List] = None
    prunen_: Optional[int] = None
    prunem_: Optional[int] = None

    # DEPRECATE sparsity_profile

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run the WANDA algorithm on the current state

        :param state: session state storing input model and calibration data
        :param kwargs: Unused, kept to conform to the parent method signature
        """
        model = state.model
        dataloader = state.data.calib

        self._infer_mask_block_size()

        # infer sparsity
        if self.owl_m is not None:
            logger.info(
                "Inferring layer-wise sparsities from "
                f"{len(dataloader) if dataloader else 0} calibration samples..."
            )
            activations = self._get_activations(model, dataloader)
            target_sparsities = self._infer_layer_sparsity(activations)
        self._validate_layerwise_sparsity()

        # infer sequential targets (formerly compressible layers)
        if self.sequential_targets is None:
            self.sequential_targets = get_no_split_params(state.model)
        if isinstance(self.sequential_targets, str):
            self.sequential_targets = [self.sequential_targets]

        # register hooks
        for name, module in state.model.named_modules():
            get_prunable_layers(self.layer)
            if getattr_chain(module, "quantization_scheme.weights", None) is not None:
                # HACK: previously, embeddings were not quantized because they were not
                # accessible by the layer compressor. For now, we manually ignore it,
                # but in the FUTURE this should be ignored by the user
                if not isinstance(module, torch.nn.Embedding):
                    post_hook = partial(self.compress_module, name)
                    self.register_hook(module, post_hook, "forward")

        # infer and run pipeline
        model_name = state.model.__class__.__name__
        input_names = state.data.calib.dataset.column_names
        unfixable_errors = (torch.OutOfMemoryError, torch._C._LinAlgError)
        try:
            run_sequential(
                state.model,
                sequential_targets,
                self.ignore,
                state.data.calib,
                propagate_error=True,
            )
            return True

        except Exception as exception:
            if isinstance(exception, torch.fx.proxy.TraceError):
                warnings.warn(f"Failed to trace {model_name} with inputs {input_names}")
            if isinstance(exception, unfixable_errors):
                raise exception

            warnings.warn("Falling back to layer_sequential pipeline")
            try:
                run_layer_sequential(
                    state.model,
                    sequential_targets,
                    state.data.calib,
                    propagate_error=True,
                )
                return True

            except Exception as exception:
                if isinstance(exception, TypeError):
                    warnings.warn(f"{model_name} fails layer-wise assumptions")
                if isinstance(exception, unfixable_errors):
                    raise exception

                warnings.warn(
                    "Falling back to basic pipeline, which requires extra memory and "
                    "may result in decreased accuracy"
                )
                run_basic(state.model, state.data.calib)
                return True

        self.apply_compression(calibration_dataloader)

        return True

    def _validate_layerwise_sparsity(self):
        if isinstance(self.sparsity, float):
            # single sparsity will be applied to all layers
            return

        target_layers = list(self.compressible_layers_.keys())

        if len(target_layers) != len(self.sparsity):
            raise ValueError(
                "Number of layer targets must match the number of "
                f"sparsities. Got {len(target_layers)} layers and "
                f"{len(self.sparsity)} sparsities"
            )

    def _pruning_arguments(self, sparsity) -> Dict[str, Any]:
        """
        Gather the parameters needed for root module compression in a dict

        :param sparsity: target sparsity
        :return: dict of params for pruning
        """
        return {
            "sparsity": sparsity,
            "prunen": self.prunen_,
            "prunem": self.prunem_,
        }

    def _infer_mask_block_size(self):
        """
        Infer the mask block size from the mask structure.
        Parses mask_structure of the form N:M where N, M are integers that
        define a custom block shape; and sets prunen_ and prunem_ accordingly.

        :post-condition: prunen_ and prunem_ are set
        """
        if self.mask_structure is None:
            raise ValueError("mask_structure must be defined")

        self.prunen_, self.prunem_ = list(map(int, self.mask_structure.split(":")))

    def _infer_layer_sparsity(self, activations):
        wanda = {}
        for name, layer in self.compressible_layers_.items():
            prunable_layers = get_prunable_layers(layer)
            z = [
                m.weight.abs() * activations[f"{name}.{n}"].unsqueeze(0)
                for n, m in prunable_layers.items()
            ]
            wanda[name] = torch.cat([item.flatten().cpu() for item in z])

        del activations
        torch.cuda.empty_cache()

        outlier_ratios = {}
        for group in wanda:
            threshold = torch.mean(wanda[group]) * self.owl_m
            outlier_ratios[group] = (
                100 * (wanda[group] > threshold).sum().item() / wanda[group].numel()
            )
        outlier_ratios_arr = np.array([outlier_ratios[k] for k in outlier_ratios])
        for k in outlier_ratios:
            outlier_ratios[k] = (outlier_ratios[k] - outlier_ratios_arr.min()) * (
                1
                / (outlier_ratios_arr.max() - outlier_ratios_arr.min())
                * self.owl_lmbda
                * 2
            )
        outlier_ratios_arr = np.array([outlier_ratios[k] for k in outlier_ratios])
        sparsities = {
            k: 1
            - (
                outlier_ratios[k]
                - np.mean(outlier_ratios_arr)
                + (1 - float(self.sparsity))
            )
            for k in outlier_ratios
        }
        logger.info(f"OWL sparsities for sp={self.sparsity} are:")
        for k in sparsities:
            logger.info(f"Sparsity for {k}: {sparsities[k]}")
        return sparsities

    def _get_wanda(self, model, dataloader, nsamples=128):
        acts = defaultdict(int)

        def save_acts(module, input, name):
            nonlocal acts

            if isinstance(input, tuple):
                input = input[0]
            acts[name] += 1.0 / nsamples * input.pow(2).sum(dim=(0, 1)).sqrt()

        hooks = [
            self.register_hook(mod, partial(save_acts, name=name), "forward_pre")
            for name, mod in self.model.named_modules()
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name
            # This is dumb. We should only attach to the modules which will be compressed
        ]
        run_basic(model, dataloader)
        self.remove_hooks(hooks)

        return acts
