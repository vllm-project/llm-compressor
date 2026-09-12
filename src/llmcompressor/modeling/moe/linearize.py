import contextlib
from functools import wraps
from typing import Type

import torch
import tqdm
from compressed_tensors.offload import get_cache_init_kwargs, load_offloaded_model
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
      The model is loaded in original 3D format. Linearization is deferred to the
      sequential pipeline for efficient per-subgraph conversion via `linearize_moe_layer`.

    For 2D checkpoints (model type with linearize mappings):
      The checkpoint is loaded directly in linearized format by registering patch mappings.
      Save conversion mappings are registered so the model can be saved in the correct
      format after pipeline operations.

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
        # linearization is deferred to pipelines
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


def linearize_moe(model: PreTrainedModel):
    """
    Linearize a mixture-of-experts model after it has been loaded. For more
    runtime-efficient loading, please see `load_quantizable_moe`.

    Experts modules will be replaced by either two pathways:
    1. The expert module has a registered replacement. This is required for
    2. The expert module conforms to the standard transformers MoE format
    (as designated by the `use_experts_implementation` decorator)

    Modules already in LinearExperts2D format are left as-is.

    :param model: model containing MoE layers to linearize
    """
    # Clear cached lookup to detect experts in the current state of the model
    if hasattr(model, "_moe_lookup"):
        delattr(model, "_moe_lookup")

    all_moes = get_non_linearized_moes(model)
    non_linearized_moes = {
        module: name for module, name in all_moes.items()
        if not isinstance(module, LinearExperts2D)
    }

    if len(non_linearized_moes) <= 0:
        return model

    logger.warning(
        "MoE is being linearized after loading in order to support efficient "
        "calibration of experts. However, this may be inefficient if the model "
        "checkpoint is already linearized (2D -> 3D -> 2D). Consider registering "
        "a load converter for faster load times. See "
        "https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support"  # noqa: E501
    )

    # If model has active offload caches, fully onload it before linearizing
    # to avoid OOM when accessing offloaded parameters
    has_offload_caches = any(
        get_cache_init_kwargs(module) for module in non_linearized_moes.keys()
    )
    if has_offload_caches:
        load_offloaded_model(model)

    for module, name in tqdm.tqdm(
        non_linearized_moes.items(), desc="Linearizing experts"
    ):
        config = getattr(module, "config", model.config)
        linear_experts_cls = LinearExperts2D.get_linear_experts_cls(module.__class__)
        # Never setup offloading here - it will be re-applied by the pipeline
        linear_moe = linear_experts_cls.from_experts_module(
            module, config, setup_offloading=False
        )
        model.set_submodule(name, linear_moe)


def get_non_linearized_moes(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]]:
    """
    Return all modules which are recognized to be experts layers. Also sets an attribute
    on the model to store the lookup.

    A module is recognized
    as an experts layer if it conforms to the `FusedExpertsProtocol` or is registered by
    `LinearExperts2D`.

    :param model: model with modules to check for experts
    :return: list of named modules which are recognized as experts layers
    """

    if not hasattr(model, "_moe_lookup"):
        model._moe_lookup = {
            module: name
            for name, module in model.named_modules()
            if isinstance(module, FusedExpertsProtocol)
            or LinearExperts2D.get_registration(module.__class__) is not None
        }
    return model._moe_lookup


def linearize_moe_layer(
    model: PreTrainedModel,
    subgraph_modules: list[torch.nn.Module],
) -> list[tuple[torch.nn.Module, dict]]:
    """
    Linearize MoE layers within a subgraph during sequential calibration.
    Offloading is deferred so calibration can run on the newly created modules before
    they are wrapped again.

    Handles both 3D experts (which need linearization) and already-linearized 2D experts
    (from checkpoints loaded via patch mappings), capturing offload kwargs for both.

    :param model: the full model, used for config fallback and set_submodule
    :param subgraph_modules: modules in the subgraph to check for experts
    :return: list of (new LinearExperts2D module, offload kwargs from original)
    """
    subgraph_set = set(subgraph_modules)
    moe_lookup = get_non_linearized_moes(model)

    non_linearized = [
        (moe_lookup[module], module) for module in subgraph_set
        if module in moe_lookup and not isinstance(module, LinearExperts2D)
    ]

    linearized = []
    for name, module in tqdm.tqdm(non_linearized, desc="Linearizing experts in subgraph"):
        offload_kwargs = get_cache_init_kwargs(module)
        config = getattr(module, "config", model.config)
        linear_experts_cls = LinearExperts2D.get_linear_experts_cls(module.__class__)
        linear_moe = linear_experts_cls.from_experts_module(
            module, config, setup_offloading=False
        )
        model.set_submodule(name, linear_moe)
        linearized.append((linear_moe, offload_kwargs))

    for _name, module in non_linearized:
        del moe_lookup[module]

    # Note: Already-linearized 2D modules (from 2D checkpoints loaded via patch mappings)
    # are left as-is. They have their offloading set up during loading and don't need
    # deferred offloading setup like the 3D->2D converted modules.

    return linearized
