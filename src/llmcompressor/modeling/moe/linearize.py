import contextlib
from functools import wraps
from typing import Type

import torch
import tqdm
from compressed_tensors.offload import get_cache_init_kwargs
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
    to set up save conversion mappings for MoE models. The model is always loaded in
    its original 3D format — linearization is deferred to the sequential pipeline
    for efficient per-subgraph conversion via `linearize_moe_layer`.

    If checkpoint conversion mappings exist for the model type, save mappings are
    registered so that the model can be saved in the correct checkpoint format after
    pipeline linearization.

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

        # load model in 3D format — linearization is deferred to the
        # pipeline for efficient per-subgraph conversion
        model = original_from_pretrained(*args, **kwargs)

        # set up save mappings so saving after pipeline linearization
        # produces the correct checkpoint key format
        if has_linearize_load_mappings(model_type):
            _experts_cls, _load_map, save_map = get_linearize_load_mappings(model_type)
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

    :param model: model containing MoE layers to linearize
    """
    non_linearized_moes = get_non_linearized_moes(model)

    if len(non_linearized_moes) <= 0:
        return model

    logger.warning(
        "MoE is being linearized after loading in order to support efficient "
        "calibration of experts. However, this may be inefficient if the model "
        "checkpoint is already linearized (2D -> 3D -> 2D). Consider registering "
        "a load converter for faster load times. See "
        "https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-moe-support"  # noqa: E501
    )

    for name, module in tqdm.tqdm(non_linearized_moes, desc="Linearizing experts"):
        config = getattr(module, "config", model.config)
        linear_experts_cls = LinearExperts2D.get_linear_experts_cls(module.__class__)
        linear_moe = linear_experts_cls.from_experts_module(module, config)
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
            module: name for name, module in model.named_modules()
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
    Offloading is deferred — the caller must set up offloading after calibration.

    :param model: the full model, used for config fallback and set_submodule
    :param subgraph_modules: modules in the subgraph to check for experts
    :return: list of (new LinearExperts2D module, offload kwargs from original)
    """
    subgraph_set = set(subgraph_modules)
    moe_lookup = model._moe_lookup

    non_linearized = [
        (moe_lookup[module], module) for module in subgraph_set if module in moe_lookup
    ]

    linearized = []
    for name, module in non_linearized:
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

    return linearized
