import warnings
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import Any

import numpy
import torch
from compressed_tensors.offload import get_execution_device
from compressed_tensors.utils import match_named_modules, match_targets
from loguru import logger
from pydantic import Field, PrivateAttr, field_validator, model_validator

from llmcompressor.core import Event, State
from llmcompressor.modifiers.modifier import Modifier
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pytorch.utils.helpers import tensors_to_device
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context
from llmcompressor.utils.pytorch.module import infer_sequential_targets

PRUNABLE_LAYER_TYPES = ["Linear", "Conv1d", "Conv2d", "Conv3d"]


class SparsityModifierBase(Modifier):
    """
    Abstract base class which implements functionality related to oneshot sparsity.
    Inheriters must implement `calibrate_module` and `compress_modules`
    """

    requires_calibration_data: bool = True

    # modifier arguments
    sparsity: float | list[float] | None
    sparsity_profile: str | None = None
    mask_structure: str = "0:0"
    owl_m: int | None = None
    owl_lmbda: float | None = None

    # data pipeline arguments
    sequential_update: bool | None = False  # deprecated
    targets: str | list[str] = ["Linear"]
    ignore: list[str] = Field(default_factory=list)

    # private variables
    _prune_n: int | None = PrivateAttr(default=None)
    _prune_m: int | None = PrivateAttr(default=None)
    _module_names: dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _module_sparsities: dict[torch.nn.Module, float] = PrivateAttr(default_factory=dict)
    _sequential_targets: list[str] = PrivateAttr(default_factory=list)
    _owl_sparsities: dict[str, float] = PrivateAttr(default_factory=dict)

    @property
    def required_pipeline(self) -> str | None:
        # OWL compares statistics across all layers before choosing any sparsities.
        return "basic" if self.sparsity_profile == "owl" else None

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
    def validate_sparsity_profile(cls, value: str | None) -> bool:
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
        args: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ):
        raise NotImplementedError()

    @abstractmethod
    def compress_modules(self):
        raise NotImplementedError()

    def on_initialize(self, state: "State", **kwargs) -> bool:
        """
        Prepare global configuration without resolving or calibrating model layers.

        :param state: session state storing input model and calibration data
        """
        if self.sparsity_profile == "owl":
            self._sequential_targets = infer_sequential_targets(
                state.model, sequential_targets=kwargs.get("sequential_targets")
            )
        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        modules = kwargs.get("modules")
        if self.sparsity_profile == "owl":
            if modules is not None:
                raise ValueError("OWL sparsity profiling requires pipeline='basic'.")
            logger.info("Using OWL to infer layer sparsities before calibration")
            layers = dict(match_named_modules(state.model, self._sequential_targets))
            self._owl_sparsities = self._infer_owl_layer_sparsity(
                state.model, layers, state.data.calib
            )
        self._register_calibration_hooks(state.model, modules)

    def on_sequential_epoch_start(
        self, state: State, event: Event, modules: list[torch.nn.Module], **kwargs
    ):
        self._register_calibration_hooks(state.model, modules)

    def _register_calibration_hooks(
        self, model: torch.nn.Module, modules: list[torch.nn.Module] | None = None
    ):
        # Resolve live modules after any pipeline replacements (e.g. MoE linearization).
        target_layers = list(match_named_modules(model, self.targets))
        if isinstance(self.sparsity, list) and len(target_layers) != len(self.sparsity):
            raise ValueError(
                f"{type(self).__name__} received {len(self.sparsity)} sparsity values, "
                f"but model has {len(target_layers)} target layers"
            )
        module_scope = None if modules is None else set(modules)
        for index, (layer_name, layer) in enumerate(target_layers):
            match self.sparsity:
                case list():
                    layer_sparsity = self.sparsity[index]
                case _:
                    layer_sparsity = self.sparsity

            for name, module in match_named_modules(layer, PRUNABLE_LAYER_TYPES):
                if module_scope is not None and module not in module_scope:
                    continue
                name = ".".join(part for part in (layer_name, name) if part)

                if match_targets(name, module, self.ignore):
                    continue

                # HACK: previously, embeddings were not quantized because they were not
                # accessible by the layer compressor. For now, we manually ignore it,
                # but in the FUTURE this should be ignored by the user
                if isinstance(module, torch.nn.Embedding):
                    continue

                if module in self._module_names:
                    continue

                if self._owl_sparsities:
                    # OWL assigns sparsity to sequential blocks, while `targets`
                    # may select individual linears within those blocks.
                    owner = next(
                        (
                            parent
                            for parent in sorted(
                                self._owl_sparsities, key=len, reverse=True
                            )
                            if not parent
                            or name == parent
                            or name.startswith(parent + ".")
                        ),
                        None,
                    )
                    if owner is None:
                        raise ValueError(f"No OWL sparsity was inferred for {name}")
                    layer_sparsity = self._owl_sparsities[owner]

                if name.endswith("lm_head"):
                    logger.warning(
                        "`lm_head` was previously auto-ignored by SparseGPT and Wanda "
                        "modifiers and is not advised. Please add `re:.*lm_head` to "
                        "your ignore list if this was unintentional"
                    )

                self._module_names[module] = name
                self._module_sparsities[module] = layer_sparsity
                self.register_hook(module, self.calibrate_module, "forward")

    def on_sequential_epoch_end(self, state: State, event: Event, **kwargs):
        self.compress_modules()

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        self.remove_hooks()

    def _infer_owl_layer_sparsity(
        self,
        model: torch.nn.Module,
        layers: dict[str, torch.nn.Module],
        dataloader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        activations = self._get_activations(model, dataloader)

        groups = {}
        for name, layer in layers.items():
            prunable_layers = dict(match_named_modules(layer, PRUNABLE_LAYER_TYPES))
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

    def _get_activations(self, model, dataloader, nsamples=128) -> dict[str, int]:
        acts = defaultdict(int)

        def save_acts(_module, input: tuple[Any, ...] | torch.Tensor, name: str):
            nonlocal acts
            if isinstance(input, tuple):
                input = input[0]
            acts[name] += 1.0 / nsamples * input.pow(2).sum(dim=(0, 1)).sqrt()

        hooks = set(
            self.register_hook(mod, partial(save_acts, name=name), "forward_pre")
            for name, mod in model.named_modules()
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name
        )
        # This is a statistics pass, not another compression epoch. Dispatching
        # lifecycle events here would recursively initialize and run the recipe.
        try:
            with (
                HooksMixin.disable_hooks(keep=hooks),
                calibration_forward_context(model),
                DisableQuantization(model),
            ):
                device = get_execution_device(model)
                for batch in dataloader:
                    model(**tensors_to_device(batch, device))
        finally:
            self.remove_hooks(hooks)

        return acts

    def _split_mask_structure(self, mask_structure: str) -> tuple[int, int]:
        n, m = mask_structure.split(":")
        return int(n), int(m)
