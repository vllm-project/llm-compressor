import contextlib
from functools import wraps
from typing import Type
from weakref import WeakKeyDictionary

import torch
import tqdm
from compressed_tensors.offload import get_execution_device
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.module import (
    remove_module_offload,
    subgraph_offload_modules,
    subgraph_onload_modules,
)
from compressed_tensors.utils import patch_attr
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedConfig,
    PreTrainedModel,
)
from transformers.conversion_mapping import (
    register_checkpoint_conversion_mapping,
)
from transformers.monkey_patching import clear_patch_mapping, register_patch_mapping

from llmcompressor.modeling.moe.helpers import FusedExpertsProtocol

from .conversion_mappings import (
    get_linearize_load_mappings,
    has_linearize_load_mappings,
    set_save_conversion_mapping,
)
from .linear_experts import LinearExperts2D


def _is_moe_module(module: torch.nn.Module) -> bool:
    # FusedExpertsProtocol and LinearExperts2D identify the outer experts
    # container. Individual ExpertMLP children do not satisfy either protocol.
    return (
        isinstance(module, FusedExpertsProtocol)
        or isinstance(module, LinearExperts2D)
        or LinearExperts2D.get_registration(module.__class__) is not None
    )


@contextlib.contextmanager
def load_quantizable_moe(model_cls: Type[PreTrainedModel] = AutoModelForCausalLM):
    """
    Context manager for loading MoE models for calibration and quantization.

    This context manager patches the `from_pretrained` method of the given model class
    to handle both 3D and 2D (linearized) MoE checkpoint formats.

    For 3D checkpoints (model type without linearize mappings):
      The model is loaded in original 3D format. Linearization is deferred
      to the sequential pipeline for efficient per-subgraph conversion via
      `linearize_moe_layer`.

    For 2D checkpoints (model type with linearize mappings):
      The checkpoint is loaded directly in linearized format by registering patch
      mappings. Save conversion mappings are registered so the model can be saved
      in the correct format after pipeline operations.

    :param model_cls: The model class to patch, defaults to AutoModelForCausalLM
    """
    original_from_pretrained = model_cls.from_pretrained
    patched_fn_called = False

    @classmethod
    @wraps(original_from_pretrained)
    def patched(cls, *args, **kwargs):
        nonlocal patched_fn_called
        patched_fn_called = True

        config = AutoConfig.from_pretrained(*args, **kwargs)
        model_type = config.model_type

        # model is 3D (or otherwise doesn't have mappings)
        # defer linearization to pipelines
        if not has_linearize_load_mappings(model_type):
            model = original_from_pretrained(*args, **kwargs)
            return model

        # prepare to load linearized weights from 2D checkpoint
        experts_cls, load_map, save_map = get_linearize_load_mappings(model_type)
        linear_experts_2d_cls = LinearExperts2D.get_linear_experts_cls(experts_cls)
        register_patch_mapping({experts_cls.__name__: linear_experts_2d_cls})
        register_checkpoint_conversion_mapping(model_type, load_map, overwrite=True)

        # load model
        model: PreTrainedModel = original_from_pretrained(*args, **kwargs)

        # prepare for saving to be called later
        clear_patch_mapping()
        set_save_conversion_mapping(model, save_map)
        register_checkpoint_conversion_mapping(model_type, save_map, overwrite=True)

        return model

    with patch_attr(model_cls, "from_pretrained", patched):
        try:
            yield
        finally:
            if not patched_fn_called:
                logger.warning(
                    f"`{model_cls.__name__}.from_pretrained` was never called. If you "
                    f"are loading with a model class other than {model_cls.__name__}, "
                    "please pass as argument to `load_quantizable_moe`"
                )


def get_moe_modules(
    model: torch.nn.Module,
    reset_cache: bool = False,
) -> WeakKeyDictionary:
    """Return MoE modules cached by identity, with qualified names as values."""

    if not hasattr(model, "_moe_lookup") or reset_cache:
        model._moe_lookup = WeakKeyDictionary(
            {
                module: name
                for name, module in model.named_modules()
                if _is_moe_module(module)
            }
        )

    return model._moe_lookup


def get_non_linearized_moes(
    model: torch.nn.Module,
    modules: dict[str, torch.nn.Module] | None = None,
) -> list[tuple[str, torch.nn.Module]]:
    """Return cached MoEs that still need linearization."""
    moe_lookup = get_moe_modules(model)
    candidates = (
        list(modules.items())
        if modules is not None
        else [(name, module) for module, name in moe_lookup.items()]
    )
    return [
        (name, module)
        for name, module in candidates
        if module in moe_lookup and not isinstance(module, LinearExperts2D)
    ]


def get_linearized_moes(
    model: torch.nn.Module,
    modules: dict[str, torch.nn.Module] | None = None,
) -> list[tuple[str, torch.nn.Module]]:
    """Return cached MoEs that are already linearized."""
    moe_lookup = get_moe_modules(model)
    candidates = (
        list(modules.items())
        if modules is not None
        else [(name, module) for module, name in moe_lookup.items()]
    )
    return [
        (name, module)
        for name, module in candidates
        if module in moe_lookup and isinstance(module, LinearExperts2D)
    ]


def get_linearized_children(
    name: str,
    module: LinearExperts2D,
    modules: dict[str, torch.nn.Module] | None = None,
) -> dict[str, torch.nn.Module]:
    """Return the named descendants of a linearized MoE for repack cleanup."""
    descendants = set(module.modules()) - {module}
    if modules is not None:
        return {
            module_name: child
            for module_name, child in modules.items()
            if child in descendants
        }

    return {
        f"{name}.{relative_name}": child
        for relative_name, child in module.named_modules()
        if relative_name
    }


def _remove_linearized_children(
    name: str,
    module: LinearExperts2D,
    modules: dict[str, torch.nn.Module],
    offload_kwargs: dict[str, dict],
) -> None:
    """Remove a linearized layer's children from bookkeeping after repacking."""
    children = get_linearized_children(name, module, modules)
    for child_name in children:
        modules.pop(child_name, None)

    root_kwargs = offload_kwargs.get(name)
    for child_name in children:
        # use the child's offload settings if we don't
        # have it for the parent
        if root_kwargs is None:
            root_kwargs = offload_kwargs.get(child_name)
        offload_kwargs.pop(child_name, None)

    if root_kwargs is not None:
        offload_kwargs[name] = root_kwargs


def _add_module_children(
    name: str,
    module: torch.nn.Module,
    modules: dict[str, torch.nn.Module],
    offload_kwargs: dict[str, dict],
) -> None:
    """Add a layer's children and inherit its root offload settings."""
    named_modules = {
        name if not relative_name else f"{name}.{relative_name}": child
        for relative_name, child in module.named_modules()
    }

    modules.update(named_modules)

    if offload_kwargs.get(name) is not None:
        offload_kwargs.update(
            {  # inherit parent offload settings for children
                module_name: offload_kwargs[name]
                for module_name in named_modules
                if module_name != name
            }
        )


def repack_moe(
    model: PreTrainedModel,
    modules: dict[str, torch.nn.Module] | None = None,
    offload_kwargs: dict[str, dict] | None = None,
) -> tuple[dict[str, torch.nn.Module] | None, dict[str, dict] | None]:
    """Repack linearized MoEs and return updated bookkeeping copies."""
    if offload_kwargs is not None and modules is None:
        raise ValueError(
            "If offload_kwargs is provided, modules must also be "
            "provided to update bookkeeping."
        )

    # offload_kwargs is passed in here for us to update bookkeeping
    # for new modules, if it is not passed in, then the model is in
    # disk offloaded format and we have to onload and offload each
    # module as we repack
    loop_offloading = True if offload_kwargs is None else False

    updated_modules = None if modules is None else modules.copy()
    updated_offload_kwargs = None if offload_kwargs is None else offload_kwargs.copy()

    moe_lookup = get_moe_modules(model)
    linearized_moes = get_linearized_moes(model, updated_modules)

    desc = "Repacking specified modules" if modules is not None else "Repacking experts"
    for i in tqdm.tqdm(range(len(linearized_moes)), desc=desc):
        # references only exist in this loop
        name, module = linearized_moes[i]

        # A parent conversion may have already replaced this module.
        # this should actually be redundant and can consider removing it
        if module not in moe_lookup:
            linearized_moes[i] = (None, None)
            continue

        layer_offload_kwargs = None
        # this guard is needed for some tests, shouldn't be 
        # necessary in practice
        should_offload = loop_offloading and isinstance(
            module._parameters, OffloadCache
        )

        if should_offload:
            layer_modules = {name: module}
            layer_modules.update(get_linearized_children(name, module, updated_modules))
            layer_offload_kwargs = subgraph_onload_modules(layer_modules)

        try:
            # generate new repacked module and replace in model
            new_module = repack_moe_layer(module)
            _replace(model, name, new_module)

            if should_offload:
                _remove_linearized_children(
                    name, module, layer_modules, layer_offload_kwargs
                )
                _add_module_children(
                    name, new_module, layer_modules, layer_offload_kwargs
                )
            elif not loop_offloading:
                _remove_linearized_children(
                    name, module, updated_modules, updated_offload_kwargs
                )
                _add_module_children(
                    name, new_module, updated_modules, updated_offload_kwargs
                )
        finally:
            if should_offload:
                subgraph_offload_modules(layer_modules, layer_offload_kwargs)

            # remove references
            linearized_moes[i] = (None, None)

    return updated_modules, updated_offload_kwargs


def repack_moe_layer(module: LinearExperts2D) -> torch.nn.Module:
    construction_device = get_execution_device(module)
    return module.to_experts_module(construction_device=construction_device)


def linearize_moe(
    model: PreTrainedModel,
    modules: dict[str, torch.nn.Module] | None = None,
    offload_kwargs: dict[str, dict] | None = None,
) -> tuple[dict[str, torch.nn.Module] | None, dict[str, dict] | None]:
    """Linearize MoEs and return updated bookkeeping copies if modules
    or offload_kwargs are provided."""
    if offload_kwargs is not None and modules is None:
        raise ValueError(
            "If offload_kwargs is provided, modules must also be "
            "provided to update bookkeeping."
        )

    # offload_kwargs is passed in here for us to update bookkeeping
    # for new modules, if it is not passed in, then the model is in
    # disk offloaded format and we have to onload and offload each
    # module as we linearize
    loop_offloading = True if offload_kwargs is None else False

    updated_modules = None if modules is None else modules.copy()
    updated_offload_kwargs = None if offload_kwargs is None else offload_kwargs.copy()

    non_linearized_moes = get_non_linearized_moes(model, updated_modules)

    if modules is None:
        logger.warning(
            "MoE is being linearized after loading in order to support efficient "
            "calibration of experts. However, this may be inefficient if the model "
            "checkpoint is already linearized (2D -> 3D -> 2D). Consider registering "
            "a load converter for faster load times. See "
            "https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support"
        )

    for i in tqdm.tqdm(range(len(non_linearized_moes)), desc="Linearizing"):
        name, module = non_linearized_moes[i]

        # Only perform the per-layer offload cycle when the source module is
        # actually offloaded. Plain in-memory modules must not acquire inferred
        # offload wrappers for parameterless descendants.
        should_offload = loop_offloading and isinstance(
            module._parameters, OffloadCache
        )
        if should_offload:
            layer_modules = {name: module}
            layer_offload_kwargs = subgraph_onload_modules(layer_modules)

        try:
            # generate new linearized module and replace in model
            new_module = linearize_moe_layer(model, module)
            _replace(model, name, new_module)

            if should_offload:
                # update the offload kwargs if we need to offload right now
                _add_module_children(
                    name, new_module, layer_modules, layer_offload_kwargs
                )
            elif not loop_offloading:
                # update bookkeeping if we are not offloading right now
                _add_module_children(
                    name, new_module, updated_modules, updated_offload_kwargs
                )
        finally:
            if should_offload:
                subgraph_offload_modules(layer_modules, layer_offload_kwargs)

            non_linearized_moes[i] = (None, None)  # remove references

    return updated_modules, updated_offload_kwargs


def linearize_moe_layer(
    model: PreTrainedModel,
    module: torch.nn.Module,
) -> LinearExperts2D:
    """Linearize a single recognized MoE layer."""
    if not isinstance(
        module, FusedExpertsProtocol
    ) and not LinearExperts2D.get_registration(module.__class__):
        raise ValueError(
            f"Module {type(module).__name__} is not a recognized MoE layer"
        )

    config = _experts_config(module, model)
    linear_experts_cls = LinearExperts2D.get_linear_experts_cls(module.__class__)
    construction_device = get_execution_device(module)
    return linear_experts_cls.from_experts_module(
        module, config, construction_device=construction_device
    )


def _experts_config(
    module: torch.nn.Module,
    model: PreTrainedModel,
) -> PreTrainedConfig:
    """Return the config expected by the native experts constructor."""
    if (config := getattr(module, "config", None)) is not None:
        return config

    config = model.config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        return get_text_config()

    return getattr(config, "text_config", config)


def _replace(
    model: PreTrainedModel,
    name: str,
    new_module: torch.nn.Module,
):
    """Replace a module in the model and update the MoE lookup."""
    old_module = model.get_submodule(name)
    model.set_submodule(name, new_module)

    if hasattr(model, "_moe_lookup"):
        # Update the MoE lookup with the new module
        moe_lookup = get_moe_modules(model)
        if old_module in moe_lookup:
            moe_lookup.pop(old_module)
            moe_lookup[new_module] = name

    # remove cache mappings before emptying the old module. Module.to_empty() walks
    # child parameters and would otherwise assign meta tensors through OffloadCache,
    # which treats those assignments as weight updates.
    for submodule in old_module.modules():
        if isinstance(submodule._parameters, OffloadCache):
            remove_module_offload(submodule, onload_tensors=False)
    old_module.to_empty(device="meta")
