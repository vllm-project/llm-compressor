import warnings
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from compressed_tensors.utils import (
    align_module_device,
    get_execution_device,
    update_offload_parameter,
)
from loguru import logger
from pydantic import PrivateAttr, field_validator, model_validator

from llmcompressor.core import State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.pruning.wanda.utils.wanda_sparsify import (
    accumulate_row_scalars,
    make_empty_row_scalars,
    sparsify_weight,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.basic import run_pipeline as run_basic
from llmcompressor.pipelines.layer_sequential import (
    run_pipeline as run_layer_sequential,
)
from llmcompressor.pipelines.sequential import run_pipeline as run_sequential
from llmcompressor.utils.metric_logging import CompressionLogger
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
    module_targets: Union[str, List[str], None] = None
    targets: Union[str, List[str], None] = None  # deprecated, clones sequential_targets
    sequential_targets: Union[str, List[str], None] = None

    # private variables
    _prune_n: Optional[int] = PrivateAttr(default=None)
    _prune_m: Optional[int] = PrivateAttr(default=None)
    _row_scalars: Dict[torch.nn.Module, torch.Tensor] = PrivateAttr(
        default_factory=dict
    )
    _num_samples: Dict[torch.nn.Module, int] = PrivateAttr(default_factory=dict)
    _update_size: Optional[int] = PrivateAttr(default=None)

    @field_validator("sparsity_profile", mode="before")
    def validate_sparsity_profile(cls, value) -> None:
        if value is not None:
            warnings.warn(
                "`sparsity_profile` is deprecated, use `owl_m` and `owl_lmbda`"
            )
        return None

    @field_validator("targets", mode="before")
    def validate_targets(cls, value) -> None:
        if value is not None:
            warnings.warn(
                "`targets` is deprecated, use `module_targets` and `sequential_targets`"
            )
        return None

    @model_validator(mode="after")
    def validate_model_after(model: "WandaPruningModifier") -> Dict[str, Any]:
        sparsity = model.sparsity
        owl_m = model.owl_m
        owl_lmbda = model.owl_lmbda
        mask_structure = model.mask_structure

        if (owl_m is not None) ^ (owl_lmbda is not None):
            raise ValueError("Must provide both `owl_m` and `owl_lmbda` or neither")

        if owl_m is not None and sparsity is not None:
            raise ValueError("Cannot provide both sparsity and owl parameters")

        model._prune_n, model._prune_m = mask_structure.split(":")

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize and run the WANDA algorithm on the current state

        :param state: session state storing input model and calibration data
        :param kwargs: Unused, kept to conform to the parent method signature
        """
        model = state.model
        dataloader = state.data.calib

        # infer module and sequential targets
        self.sequential_targets = self._infer_sequential_targets(model)

        # infer layer sparsities
        if self.owl_m is not None and self.owl_lmbda is not None:
            logger.info(
                "Using OWL to infer target layer-wise sparsities from "
                f"{len(dataloader) if dataloader else 0} calibration samples..."
            )
            self.sparsity = self._infer_owl_layer_sparsity()

        # infer update size
        if self._update_size is None:
            self._update_size = len(dataloader)

        # register hooks
        for index, name, layer in enumerate(get_layers(self.sequential_targets, model)):
            if isinstance(self.sparsity, Dict):
                layer_sparsity = self.sparsity[name]
            elif isinstance(self.sparsity, List):
                layer_sparsity = self.sparsity[index]
            else:
                layer_sparsity = self.sparsity

            for name, module in get_prunable_layers(layer):
                post_hook = partial(
                    self.compress_module,
                    name,
                    layer_sparsity,
                    self._prune_n,
                    self._prune_m,
                )
                self.register_hook(module, post_hook, "forward")

        # infer and run pipeline
        model_name = state.model.__class__.__name__
        input_names = state.data.calib.dataset.column_names
        unfixable_errors = (torch.OutOfMemoryError, torch._C._LinAlgError)
        try:
            run_sequential(
                state.model,
                self.sequential_targets,
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
                    self.sequential_targets,
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

    def compress_module(
        self,
        name: str,
        sparsity: float,
        prune_n: int,
        prune_m: int,
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

        # After enough samples are accumulated, perform quantization
        if self._num_samples[module] >= self._update_size:
            logger.info(f"Sparsifying {name} using {self._num_samples[module]} samples")
            with (
                torch.no_grad(),
                align_module_device(module),
                CompressionLogger(module),
            ):
                sparsified_weight = sparsify_weight(
                    module=module,
                    row_scalars_dict=self._row_scalars,
                    sparsity=sparsity,
                    prune_n=prune_n,
                    prune_m=prune_m,
                )

            update_offload_parameter(module, "weight", sparsified_weight)

            # self._hessians[module] already deleted by quantize_weight
            del self._num_samples[module]

    def _infer_sequential_targets(self, model):
        if self.sequential_targets is None:
            return get_no_split_params(model)
        if isinstance(self.sequential_targets, str):
            return [self.sequential_targets]

    def _infer_owl_layer_sparsity(
        self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        activations = self._get_activations(model, dataloader)

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

    def _get_activations(self, model, dataloader, nsamples=128):
        acts = defaultdict(int)

        def save_acts(module, input, name):
            nonlocal acts
            if isinstance(input, tuple):
                input = input[0]
            acts[name] += 1.0 / nsamples * input.pow(2).sum(dim=(0, 1)).sqrt()

        # TODO: only add hooks to target modules
        hooks = set(
            self.register_hook(mod, partial(save_acts, name=name), "forward_pre")
            for name, mod in self.model.named_modules()
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name
        )
        with HooksMixin.disable_hooks(keep=hooks):
            run_basic(model, dataloader)
        self.remove_hooks(hooks)

        return acts
