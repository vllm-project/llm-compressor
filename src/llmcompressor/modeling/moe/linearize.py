import contextlib
from functools import wraps
from typing import Type
from weakref import WeakKeyDictionary

import torch
import tqdm
from compressed_tensors.offload.module import (
    subgraph_offload_modules,
    subgraph_onload_modules,
)
from compressed_tensors.utils import patch_attr
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
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
) -> WeakKeyDictionary:
    """
    Return all modules which are recognized to be experts layers.
    Includes both 3D experts (which need linearization) and
    already-linearized 2D experts. This lookup is used by the
    repack_moe and linearize_moe functions to determine
    which modules to operate on.
    """

    if hasattr(model, "_moe_lookup"):
        delattr(model, "_moe_lookup")

    model._moe_lookup = WeakKeyDictionary(
        {
            module: name
            for name, module in model.named_modules()
            if isinstance(module, FusedExpertsProtocol)
            or isinstance(module, LinearExperts2D)
            or LinearExperts2D.get_registration(module.__class__) is not None
        }
    )

    return model._moe_lookup


def get_non_linearized_moes(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    """Return recognized MoE modules that still need linearization."""
    moe_lookup = get_moe_modules(model)
    return [
        (moe_lookup[module], module)
        for module in model.modules()
        if module in moe_lookup and not isinstance(module, LinearExperts2D)
    ]


def _get_subgraph_modules(
    subgraph_modules: dict[str, torch.nn.Module],
) -> set[torch.nn.Module]:
    """Return subgraph modules and all of their descendants."""
    return {
        submodule
        for module in subgraph_modules.values()
        for submodule in module.modules()
    }


def _get_named_modules(
    name: str, module: torch.nn.Module
) -> dict[str, torch.nn.Module]:
    return {
        name if not relative_name else f"{name}.{relative_name}": child
        for relative_name, child in module.named_modules()
    }


def repack_moe(
    model: PreTrainedModel,
    subgraph_modules: dict[str, torch.nn.Module] | None = None,
    onload_and_offload: bool = False,
    offload_kwargs: dict[str, dict] | None = None,
) -> None:
    """Repack linearized MoE modules in a model or subgraph."""
    subgraph_set = (
        _get_subgraph_modules(subgraph_modules)
        if subgraph_modules is not None
        else None
    )
    moe_lookup = get_moe_modules(model)
    linearized = [
        (moe_lookup[module], module)
        for module in (subgraph_set if subgraph_set is not None else model.modules())
        if module in moe_lookup and isinstance(module, LinearExperts2D)
    ]

    desc = "Repacking experts in subgraph" if subgraph_modules else "Repacking experts"
    for name, module in tqdm.tqdm(linearized, desc=desc):
        module_dict = subgraph_modules
        layer_offload_kwargs = offload_kwargs
        if onload_and_offload:
            module_dict = _get_named_modules(name, module)
            layer_offload_kwargs = subgraph_onload_modules(module_dict)
        try:
            repack_moe_layer(model, name, module, module_dict, layer_offload_kwargs)
        finally:
            if onload_and_offload and layer_offload_kwargs:
                subgraph_offload_modules(module_dict, layer_offload_kwargs)


def repack_moe_layer(
    model: PreTrainedModel,
    name: str,
    module: LinearExperts2D,
    subgraph_modules: dict[str, torch.nn.Module] | None = None,
    offload_kwargs: dict[str, dict] | None = None,
) -> None:
    fused = module.to_experts_module()
    _replace(model, name, module, fused, subgraph_modules)

    if subgraph_modules is not None:
        for child_name in list(subgraph_modules):
            if child_name.startswith(f"{name}."):
                del subgraph_modules[child_name]

    if offload_kwargs is not None:
        for child_name in list(offload_kwargs):
            if child_name.startswith(f"{name}."):
                offload_kwargs.setdefault(name, offload_kwargs[child_name])
                del offload_kwargs[child_name]


def linearize_moe(
    model: PreTrainedModel,
    subgraph_modules: dict[str, torch.nn.Module] | None = None,
    onload_and_offload: bool = False,
    offload_kwargs: dict[str, dict] | None = None,
) -> None:
    """Linearize recognized non-linearized MoE modules in a model or subgraph."""
    subgraph_set = (
        _get_subgraph_modules(subgraph_modules)
        if subgraph_modules is not None
        else None
    )
    moe_lookup = get_moe_modules(model)
    non_linearized = [
        (moe_lookup[module], module)
        for module in (subgraph_set if subgraph_set is not None else model.modules())
        if module in moe_lookup and not isinstance(module, LinearExperts2D)
    ]

    if subgraph_modules is None:
        logger.warning(
            "MoE is being linearized after loading in order to support efficient "
            "calibration of experts. However, this may be inefficient if the model "
            "checkpoint is already linearized (2D -> 3D -> 2D). Consider registering "
            "a load converter for faster load times. See "
            "https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support"
        )

    desc = (
        "Linearizing experts in subgraph" if subgraph_modules else "Linearizing experts"
    )
    for name, module in tqdm.tqdm(non_linearized, desc=desc):
        module_dict = subgraph_modules
        layer_offload_kwargs = offload_kwargs
        if onload_and_offload:
            module_dict = _get_named_modules(name, module)
            layer_offload_kwargs = subgraph_onload_modules(module_dict)
        try:
            linearize_moe_layer(model, name, module, module_dict, layer_offload_kwargs)
        finally:
            if onload_and_offload and layer_offload_kwargs:
                subgraph_offload_modules(module_dict, layer_offload_kwargs)


def linearize_moe_layer(
    model: PreTrainedModel,
    name: str,
    module: torch.nn.Module,
    subgraph_modules: dict[str, torch.nn.Module] | None = None,
    offload_kwargs: dict[str, dict] | None = None,
) -> None:
    """Linearize a single recognized MoE layer."""
    if not isinstance(
        module, FusedExpertsProtocol
    ) and not LinearExperts2D.get_registration(module.__class__):
        raise ValueError(f"Module {name} is not a recognized MoE layer")

    config = getattr(module, "config", model.config)
    linear_experts_cls = LinearExperts2D.get_linear_experts_cls(module.__class__)
    linear_moe = linear_experts_cls.from_experts_module(module, config)
    _replace(model, name, module, linear_moe, subgraph_modules)

    if subgraph_modules is not None:
        subgraph_modules.update(
            {
                f"{name}.{relative_name}": child
                for relative_name, child in linear_moe.named_modules()
                if relative_name
            }
        )

    if offload_kwargs is not None and name in offload_kwargs:
        offload_kwargs.update(
            {
                f"{name}.{relative_name}": offload_kwargs[name]
                for relative_name, _ in linear_moe.named_modules()
                if relative_name
            }
        )


def _replace(
    model: PreTrainedModel,
    name: str,
    old_module: torch.nn.Module,
    new_module: torch.nn.Module,
    module_dict: dict[str, torch.nn.Module] | None = None,
):
    """Replace a module and keep offload/subgraph bookkeeping consistent."""
    # Conversion creates a new cache with the replacement module's parameter schema.
    # Reusing the old cache would associate the wrong names and tensor layouts.
    if module_dict is not None:
        if name in module_dict and module_dict[name] is not old_module:
            raise ValueError(
                f"Module {name} in module_dict does not match the old_module "
                "being replaced. Something went very wrong."
            )
        module_dict[name] = new_module

    model.set_submodule(name, new_module)

    # Sequential tracing retains replaced modules through its bookkeeping. Release
    # their storage there, while leaving direct callers' old module references usable.
    if module_dict is not None and not hasattr(
        old_module._parameters, "offloaded_values"
    ):
        old_module.to_empty(device="meta")
