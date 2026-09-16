import contextlib
from functools import wraps
from typing import Type

import torch
import tqdm
from compressed_tensors.offload import (
    disable_offloading,
    get_cache_init_kwargs,
    offload_module,
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
from llmcompressor.pipelines.sequential.offloading import (
    disable_offloading_controlled,
)

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
) -> list[tuple[str, torch.nn.Module]]:
    """
    Return all modules which are recognized to be experts layers. 
    Includes both 3D experts (which need linearization) and 
    already-linearized 2D experts. This lookup is used by the 
    repack_moe_* and linearize_moe_* functions to determine
    which modules to operate on.
    """

    if hasattr(model, "_moe_lookup"):
        delattr(model, "_moe_lookup")

    model._moe_lookup = {
        module: name
        for name, module in model.named_modules()
        if isinstance(module, FusedExpertsProtocol)
        or isinstance(module, LinearExperts2D)
    }

    return model._moe_lookup

def repack_moe_model(model: PreTrainedModel) -> None:
    """
    Explicitly pack linearized :class:`LinearExperts2D` modules back into native
    fused 3D expert modules.

    Call this after calibration/quantization and before ``save_pretrained`` when
    the target Transformers architecture expects packed expert weights (e.g.
    ``qwen3_vl_moe``, ``qwen3_5_moe``). See
    https://github.com/vllm-project/llm-compressor/issues/2699

    :param model: model containing linearized MoE layers to repack
    :return: the same model with fused expert modules restored
    """
    moe_lookup = get_moe_linear_status(model)

    linearized = [
        (moe_lookup[module], module)
        for module in model.modules()
        if module in moe_lookup and isinstance(module, LinearExperts2D)
    ]

    # Use range because we want to avoid creating references to the 
    # modules in the list, which would prevent them from being deleted
    for i in tqdm.tqdm(range(len(linearized)), desc="Repacking experts"):
        with disable_offloading_controlled(linearized[i][1]):
            repack_moe_layer(model, linearized[i][0], linearized[i][1])

        linearized[i] = None  # remove reference to module to allow deletion

def repack_moe_subgraph(
    model: PreTrainedModel,
    subgraph_modules: list[torch.nn.Module],
) -> None:
    """
    Repack linearized :class:`LinearExperts2D` modules back into native fused 3D expert
    modules for a subgraph during sequential calibration.

    :param model: the full model, used for config fallback and set_submodule
    :param subgraph_modules: modules in the subgraph to check for experts
    :return: the same model with fused expert modules restored for the subgraph
    """
    subgraph_set = set(subgraph_modules)
    moe_lookup = get_moe_linear_status(model)
    linearized = [
        (moe_lookup[module], module)
        for module in subgraph_set
        if module in moe_lookup and isinstance(module, LinearExperts2D)
    ]

    for i in tqdm.tqdm(range(len(linearized)), desc="Repacking experts in subgraph"):
        repack_moe_layer(model, linearized[i][0], linearized[i][1])

def repack_moe_layer(
    model: PreTrainedModel, name: str, module: LinearExperts2D
) -> None:
    """
    Repack a single linearized :class:`LinearExperts2D` module back into its native
    fused 3D expert module. 
    """
    fused = module.to_experts_module()
    model.set_submodule(name, fused)

    module._onload_wrapper.replace_with(fused)

    if hasattr(model, "_moe_lookup"):
        # update the lookup to reflect the new module
        del model._moe_lookup[module]
        model._moe_lookup[fused] = name

def linearize_moe_model(model: PreTrainedModel) -> None:
    """
    Experts modules will be replaced by either two pathways:
    1. The expert module has a registered replacement. This is required for
    2. The expert module conforms to the standard transformers MoE format
    (as designated by the `use_experts_implementation` decorator)

    Modules already in LinearExperts2D format are left as-is.

    :param model: model containing MoE layers to linearize
    """

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
        "https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support"  # noqa: E501
    )

    # This is also sequential
    for i in tqdm.tqdm(range(len(non_linearized)), desc="Linearizing experts"):
        with disable_offloading_controlled(non_linearized[i][1]):
            linearize_moe_layer(model, non_linearized[i][0], non_linearized[i][1])

        non_linearized[i] = None

def linearize_moe_subgraph(
    model: PreTrainedModel,
    subgraph_modules: list[torch.nn.Module],
) -> None:
    """
    Linearize MoE layers within a subgraph during sequential calibration.
    Offloading is deferred so calibration can run on the newly created modules before
    they are wrapped again.

    Handles both 3D experts (which need linearization) and already-linearized 2D experts
    (from checkpoints loaded via patch mappings), capturing offload kwargs for both.

    :param model: the full model, used for config fallback and set_submodule
    :param subgraph_modules: modules in the subgraph to check for experts
    """
    subgraph_set = set(subgraph_modules)
    moe_lookup = get_moe_linear_status(model)

    non_linearized = [
        (moe_lookup[module], module)
        for module in subgraph_set
        if module in moe_lookup and not isinstance(module, LinearExperts2D)
    ]

    for i in tqdm.tqdm(range(len(non_linearized)), desc="Linearizing experts in subgraph"):
        linearize_moe_layer(model, non_linearized[i][0], non_linearized[i][1])

def linearize_moe_layer(
    model: PreTrainedModel, 
    name: str, 
    module: torch.nn.Module
) -> None:
    """Linearize a single module within the model"""

    if not isinstance(module, FusedExpertsProtocol) and not LinearExperts2D.get_registration(module.__class__):
        raise ValueError(f"Module {name} is not a recognized MoE layer")

    config = getattr(module, "config", model.config)
    linear_experts_cls = LinearExperts2D.get_linear_experts_cls(module.__class__)
    linear_moe = linear_experts_cls.from_experts_module(
        module, config
    )
    model.set_submodule(name, linear_moe)

    module._onload_wrapper.replace_with(linear_moe)

    if hasattr(model, "_moe_lookup"):
        del model._moe_lookup[module]
        model._moe_lookup[linear_moe] = name
