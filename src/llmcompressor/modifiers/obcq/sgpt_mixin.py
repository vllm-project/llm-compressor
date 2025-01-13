import warnings
from collections import defaultdict
from functools import partial
from typing import List, Tuple, Union

import numpy as np
import torch
from loguru import logger
from pydantic import field_validator, model_validator

from llmcompressor.core import State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.basic import run_pipeline as run_basic
from llmcompressor.pipelines.layer_sequential import (
    run_pipeline as run_layer_sequential,
)
from llmcompressor.pipelines.sequential import run_pipeline as run_sequential
from llmcompressor.utils.pytorch.module import (
    get_layers,
    get_no_split_params,
    get_prunable_layers,
    match_class,
    match_targets,
)


class SparsityModifierMixin(HooksMixin):
    @field_validator("sequential_update", mode="before", check_fields=False)
    def validate_sequential_update(cls, value: bool) -> bool:
        if not value:
            warnings.warn(
                "`sequential_update=False` is no longer supported, setting "
                "sequential_update=True",
                DeprecationWarning,
            )

        return True

    @field_validator("sparsity_profile", mode="before", check_fields=False)
    def validate_sparsity_profile(cls, value) -> None:
        if value is not None:
            warnings.warn(
                "`sparsity_profile` is deprecated, use `owl_m` and `owl_lmbda`"
            )
        return None

    @model_validator(mode="after")
    def validate_model_after(model: "Modifier") -> "Modifier":
        sparsity = model.sparsity
        owl_m = model.owl_m
        owl_lmbda = model.owl_lmbda
        mask_structure = model.mask_structure
        targets = model.targets
        sequential_targets = model.sequential_targets

        if (owl_m is not None) ^ (owl_lmbda is not None):
            raise ValueError("Must provide both `owl_m` and `owl_lmbda` or neither")

        if owl_m is not None and sparsity is not None:
            raise ValueError("Cannot provide both sparsity and owl parameters")

        if targets is not None:
            warnings.warn(
                "`targets` is deprecated, use `module_targets` and `sequential_targets`"
            )
            if sequential_targets is not None:
                raise ValueError("Cannot use both `targets` and `sequential_targets`")
            model.sequential_targets = targets
            model.targets = None

        model._prune_n, model._prune_m = model._split_mask_structure(mask_structure)

        return model

    def on_initialize(self, state: "State", **kwargs) -> bool:
        """
        Initialize and run the OBCQ algorithm on the current state

        :param state: session state storing input model and calibration data
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

        # register hooks
        for index, (name, layer) in enumerate(
            get_layers(self.sequential_targets, model).items()
        ):
            if isinstance(self.sparsity, dict):
                layer_sparsity = self.sparsity[name]
            elif isinstance(self.sparsity, list):
                layer_sparsity = self.sparsity[index]
            else:
                layer_sparsity = self.sparsity

            for name, module in layer.named_modules():
                if (
                    match_targets(module, self.module_targets)[0]
                    or match_class(module, self.module_targets)[0]
                ):
                    self._module_names[module] = name
                    self._module_sparsities[module] = layer_sparsity
                    self.register_hook(module, self.calibrate_module, "forward")

        # infer and run pipeline
        model_name = state.model.__class__.__name__
        input_names = state.data.calib.dataset.column_names
        unfixable_errors = (torch.OutOfMemoryError, torch._C._LinAlgError)
        try:
            run_sequential(
                state.model,
                state.data.calib,
                self.sequential_targets,
                self.ignore,
                self,
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
                    state.data.calib,
                    self.sequential_targets,
                    self,
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
                run_basic(state.model, state.data.calib, self)
                return True

        return True

    def _infer_sequential_targets(
        self, model: torch.nn.Module
    ) -> Union[str, List[str]]:
        if self.sequential_targets is None:
            return get_no_split_params(model)
        if isinstance(self.sequential_targets, str):
            return [self.sequential_targets]
        return self.sequential_targets

    def _infer_owl_layer_sparsity(self, activations):
        groups = {}
        for name, layer in self.compressible_layers_.items():
            prunable_layers = get_prunable_layers(layer)
            z = [
                m.weight.abs() * activations[f"{name}.{n}"].unsqueeze(0)
                for n, m in prunable_layers.items()
            ]
            groups[name] = torch.cat([item.flatten().cpu() for item in z])

        del activations
        torch.cuda.empty_cache()

        outlier_ratios = {}
        for group in groups:
            threshold = torch.mean(groups[group]) * self.owl_m
            outlier_ratios[group] = (
                100 * (groups[group] > threshold).sum().item() / groups[group].numel()
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

        def save_acts(_module, input, name):
            nonlocal acts
            if isinstance(input, tuple):
                input = input[0]
            acts[name] += 1.0 / nsamples * input.pow(2).sum(dim=(0, 1)).sqrt()

        # TODO: only add hooks to target modules
        hooks = set(
            self.register_hook(mod, partial(save_acts, name=name), "forward_pre")
            for name, mod in model.named_modules()
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name
        )
        with HooksMixin.disable_hooks(keep=hooks):
            run_basic(model, dataloader)
        self.remove_hooks(hooks)

        return acts

    def _split_mask_structure(self, mask_structure: str) -> Tuple[int, int]:
        n, m = mask_structure.split(":")
        return int(n), int(m)
