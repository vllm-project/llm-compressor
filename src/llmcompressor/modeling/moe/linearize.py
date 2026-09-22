import contextlib
from functools import wraps
from typing import Type
from weakref import WeakKeyDictionary

import torch
import tqdm
from compressed_tensors.offload.cache import OffloadCache
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


def get_moe_linear_status(
    model: torch.nn.Module,
) -> WeakKeyDictionary:
    """
    Return all modules which are recognized to be experts layers.
    Includes both 3D experts (which need linearization) and
    already-linearized 2D experts. This lookup is used by the
    repack_moe_* and linearize_moe_* functions to determine
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


def repack_moe_model(model: PreTrainedModel) -> None:
    """Repack all linearized MoE modules in a model."""
    moe_lookup = get_moe_linear_status(model)
    linearized = [
        (moe_lookup[module], module)
        for module in model.modules()
        if module in moe_lookup and isinstance(module, LinearExperts2D)
    ]

    for name, module in tqdm.tqdm(linearized, desc="Repacking experts"):
        module_dict = {name: module}
        offload_kwargs = subgraph_onload_modules(module_dict)
        try:
            repack_moe_layer(model, name, module, module_dict)
        finally:
            if offload_kwargs:
                subgraph_offload_modules(module_dict, offload_kwargs)


def repack_moe_subgraph(
    model: PreTrainedModel,
    subgraph_modules: dict[str, torch.nn.Module],
) -> None:
    """Repack linearized MoE modules contained in a subgraph."""
    subgraph_set = set(subgraph_modules.values())
    moe_lookup = get_moe_linear_status(model)
    linearized = [
        (moe_lookup[module], module)
        for module in subgraph_set
        if module in moe_lookup and isinstance(module, LinearExperts2D)
    ]

    for name, module in tqdm.tqdm(linearized, desc="Repacking experts in subgraph"):
        repack_moe_layer(model, name, module, subgraph_modules)


def repack_moe_layer(
    model: PreTrainedModel,
    name: str,
    module: LinearExperts2D,
    subgraph_modules: dict[str, torch.nn.Module] | None = None,
) -> None:
    fused = module.to_experts_module()
    _replace(model, name, module, fused, subgraph_modules)


def linearize_moe_model(model: PreTrainedModel) -> None:
    """Linearize all recognized non-linearized MoE modules in a model."""
    moe_lookup = get_moe_linear_status(model)
    non_linearized = [
        (moe_lookup[module], module)
        for module in model.modules()
        if module in moe_lookup and not isinstance(module, LinearExperts2D)
    ]

    logger.warning(
        "MoE is being linearized after loading in order to support efficient "
        "calibration of experts. However, this may be inefficient if the model "
        "checkpoint is already linearized (2D -> 3D -> 2D). Consider registering "
        "a load converter for faster load times. See "
        "https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support"
    )

    for name, module in tqdm.tqdm(non_linearized, desc="Linearizing experts"):
        module_dict = {name: module}
        offload_kwargs = subgraph_onload_modules(module_dict)
        try:
            linearize_moe_layer(model, name, module, module_dict)
        finally:
            if offload_kwargs:
                subgraph_offload_modules(module_dict, offload_kwargs)


def linearize_moe_subgraph(
    model: PreTrainedModel,
    subgraph_modules: dict[str, torch.nn.Module],
) -> None:
    """Linearize recognized non-linearized MoE modules in a subgraph."""
    subgraph_set = set(subgraph_modules.values())
    moe_lookup = get_moe_linear_status(model)
    non_linearized = [
        (moe_lookup[module], module)
        for module in subgraph_set
        if module in moe_lookup and not isinstance(module, LinearExperts2D)
    ]

    for name, module in tqdm.tqdm(
        non_linearized, desc="Linearizing experts in subgraph"
    ):
        linearize_moe_layer(model, name, module, subgraph_modules)


def linearize_moe_layer(
    model: PreTrainedModel,
    name: str,
    module: torch.nn.Module,
    subgraph_modules: dict[str, torch.nn.Module] | None = None,
) -> None:
    """Linearize a single recognized MoE module."""
    if not isinstance(module, FusedExpertsProtocol) and not LinearExperts2D.get_registration(
        module.__class__
    ):
        raise ValueError(f"Module {name} is not a recognized MoE layer")

    config = getattr(module, "config", model.config)
    linear_experts_cls = LinearExperts2D.get_linear_experts_cls(module.__class__)
    linear_moe = linear_experts_cls.from_experts_module(module, config)
    _replace(model, name, module, linear_moe, subgraph_modules)


def _replace(
    model: PreTrainedModel,
    name: str,
    old_module: torch.nn.Module,
    new_module: torch.nn.Module,
    module_dict: dict[str, torch.nn.Module] | None = None,
):
    """Replace a module and keep offload/subgraph bookkeeping consistent."""
    if isinstance(old_module._parameters, OffloadCache):
        new_module._parameters = old_module._parameters
        new_module._buffers = old_module._buffers

    if module_dict is not None:
        if module_dict.get(name) is not old_module:
            raise ValueError(
                f"Module {name} in module_dict does not match the old_module "
                "being replaced. Something went very wrong."
            )
        module_dict[name] = new_module

    model.set_submodule(name, new_module)


# Backward-compatible names used by existing model integrations.
linearize_moe = linearize_moe_model
repack_moe = repack_moe_model
