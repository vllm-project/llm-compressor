import warnings
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
import torch
from loguru import logger
from pydantic import Field, PrivateAttr, field_validator, model_validator

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.modifier import Modifier
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.utils.pytorch.module import (
    get_layers,
    get_no_split_params,
    get_prunable_layers,
    match_targets,
)


class SparsityModifierBase(Modifier):
    """
    Abstract base class which implements functionality related to oneshot sparsity.
    Inheriters must implement `calibrate_module` and `compress_modules`
    """

    # modifier arguments
    sparsity: Optional[Union[float, List[float]]]
    sparsity_profile: Optional[str] = None
    mask_structure: str = "0:0"
    owl_m: Optional[int] = None
    owl_lmbda: Optional[float] = None

    # data pipeline arguments
    sequential_update: Optional[bool] = False  # deprecated
    sequential_targets: Union[str, List[str], None] = None
    targets: Union[str, List[str]] = ["Linear"]
    ignore: List[str] = Field(default_factory=list)

    # private variables
    _prune_n: Optional[int] = PrivateAttr(default=None)
    _prune_m: Optional[int] = PrivateAttr(default=None)
    _module_names: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _target_layers: Dict[str, torch.nn.Module] = PrivateAttr(default_factory=dict)
    _module_sparsities: Dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)

    @field_validator("sequential_update", mode="before")
    def validate_sequential_update(cls, value: bool) -> bool:
        if not value:
            warnings.warn(
                "`sequential_update=False` is no longer supported, setting "
                "sequential_update=True",
                DeprecationWarning,
            )

        return True

    @field_validator("sparsity_profile", mode="before")
    def validate_sparsity_profile(cls, value: Optional[str]) -> bool:
        if value is None:
            return value

        value = value.lower()

        profile_options = ["owl"]
        if value not in profile_options:
            raise ValueError(f"Please choose profile from {profile_options}")

        return value

    @model_validator(mode="after")
    def validate_model_after(model: "SparsityModifierBase") -> "SparsityModifierBase":
        profile = model.sparsity_profile
        owl_m = model.owl_m
        owl_lmbda = model.owl_lmbda
        mask_structure = model.mask_structure

        has_owl_m = owl_m is not None
        has_owl_lmbda = owl_lmbda is not None
        has_owl = profile == "owl"
        owl_args = (has_owl_m, has_owl_lmbda, has_owl)
        if any(owl_args) and not all(owl_args):
            raise ValueError(
                'Must provide all of `profile="owl"`, `owl_m` and `owl_lmbda` or none'
            )

        model._prune_n, model._prune_m = model._split_mask_structure(mask_structure)

        return model

    @abstractmethod
    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ):
        raise NotImplementedError()

    @abstractmethod
    def compress_modules(self):
        raise NotImplementedError()

    def on_initialize(self, state: "State", **kwargs) -> bool:
        """
        Initialize and run the SparseGPT algorithm on the current state

        :param state: session state storing input model and calibration data
        """
        model: torch.nn.Module = state.model
        dataloader: torch.utils.data.DataLoader = state.data.calib

        # infer module and sequential targets
        self.sequential_targets = self._infer_sequential_targets(model)
        layers = get_layers(self.sequential_targets, model)
        self._target_layers = get_layers(
            self.targets, model
        )  # layers containing targets

        # infer layer sparsities
        if self.sparsity_profile == "owl":
            logger.info(
                "Using OWL to infer target layer-wise sparsities from "
                f"{len(dataloader) if dataloader else 0} calibration samples..."
            )
            self.sparsity = self._infer_owl_layer_sparsity(model, layers, dataloader)

        # get layers and validate sparsity
        if isinstance(self.sparsity, (list, dict)) and len(self._target_layers) != len(
            self.sparsity
        ):
            raise ValueError(
                f"{self.__repr_name__} was initialized with {len(self.sparsity)} "
                f"sparsities values, but model has {len(layers)} target layers"
            )

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        self.started_ = True

        # register hooks
        for index, (layer_name, layer) in enumerate(self._target_layers.items()):
            if isinstance(self.sparsity, dict):
                layer_sparsity = self.sparsity[layer_name]
            elif isinstance(self.sparsity, list):
                layer_sparsity = self.sparsity[index]
            else:
                layer_sparsity = self.sparsity

            for name, module in get_prunable_layers(layer).items():
                name = f"{layer_name}.{name}"

                if match_targets(name, self.ignore)[0]:
                    continue

                # HACK: previously, embeddings were not quantized because they were not
                # accessible by the layer compressor. For now, we manually ignore it,
                # but in the FUTURE this should be ignored by the user
                if isinstance(module, torch.nn.Embedding):
                    continue

                if name.endswith("lm_head"):
                    logger.warning(
                        "`lm_head` was previously auto-ignored by SparseGPT and Wanda "
                        "modifiers and is not advised. Please add `re:.*lm_head` to "
                        "your ignore list if this was unintentional"
                    )

                self._module_names[module] = name
                self._module_sparsities[module] = layer_sparsity
                self.register_hook(module, self.calibrate_module, "forward")

    def on_event(self, state: State, event: Event, **kwargs):
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, None)

        if event.type_ == EventType.SEQUENTIAL_EPOCH_END:
            self.compress_modules()

        if event.type_ == EventType.CALIBRATION_EPOCH_END:
            self.compress_modules()

            if not self.ended_:
                self.on_end(state, None)

    def on_end(self, state: State, event: Event, **kwargs):
        self.ended_ = True
        self.remove_hooks()

    def _infer_sequential_targets(
        self, model: torch.nn.Module
    ) -> Union[str, List[str]]:
        if self.sequential_targets is None:
            return get_no_split_params(model)
        if isinstance(self.sequential_targets, str):
            return [self.sequential_targets]
        return self.sequential_targets

    def _infer_owl_layer_sparsity(
        self,
        model: torch.nn.Module,
        layers: Dict[str, torch.nn.Module],
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        activations = self._get_activations(model, dataloader)

        groups = {}
        for name, layer in layers.items():
            prunable_layers = get_prunable_layers(layer)
            z = [
                m.weight.abs() * activations[f"{name}.{n}"].unsqueeze(0)
                for n, m in prunable_layers.items()
            ]
            groups[name] = torch.cat([item.flatten().cpu() for item in z])

        del activations

        outlier_ratios = {}
        for group in groups:
            threshold = torch.mean(groups[group]) * self.owl_m
            outlier_ratios[group] = (
                100 * (groups[group] > threshold).sum().item() / groups[group].numel()
            )
        outlier_ratios_arr = numpy.array([outlier_ratios[k] for k in outlier_ratios])
        for k in outlier_ratios:
            outlier_ratios[k] = (outlier_ratios[k] - outlier_ratios_arr.min()) * (
                1
                / (outlier_ratios_arr.max() - outlier_ratios_arr.min())
                * self.owl_lmbda
                * 2
            )
        outlier_ratios_arr = numpy.array([outlier_ratios[k] for k in outlier_ratios])
        sparsities = {
            k: 1
            - (
                outlier_ratios[k]
                - numpy.mean(outlier_ratios_arr)
                + (1 - float(self.sparsity))
            )
            for k in outlier_ratios
        }
        logger.info(f"OWL sparsities for sp={self.sparsity} are:")
        for k in sparsities:
            logger.info(f"Sparsity for {k}: {sparsities[k]}")
        return sparsities

    def _get_activations(self, model, dataloader, nsamples=128) -> Dict[str, int]:
        from llmcompressor.pipelines.basic import run_calibration

        acts = defaultdict(int)

        def save_acts(_module, input: Union[Tuple[Any, ...], torch.Tensor], name: str):
            nonlocal acts
            if isinstance(input, tuple):
                input = input[0]
            acts[name] += 1.0 / nsamples * input.pow(2).sum(dim=(0, 1)).sqrt()

        hooks = set(
            self.register_hook(mod, partial(save_acts, name=name), "forward_pre")
            for name, mod in model.named_modules()
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name
        )
        with HooksMixin.disable_hooks(keep=hooks):
            run_calibration(model, dataloader)
        self.remove_hooks(hooks)

        return acts

    def _split_mask_structure(self, mask_structure: str) -> Tuple[int, int]:
        n, m = mask_structure.split(":")
        return int(n), int(m)
