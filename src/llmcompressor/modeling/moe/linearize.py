import contextlib
from functools import wraps
from typing import Type

import torch
import tqdm
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
    Context manager for loading MoE models with linearized experts for
    efficient calibration and quantization.

    This context manager patches the `from_pretrained` method of the given model class
    to automatically linearize MoE (Mixture-of-Experts) layers during model loading.
    Linearization converts 3D expert weight tensors into 2D format, enabling more
    efficient calibration and quantization of individual experts.

    Two loading pathways are supported:
    1. Direct loading: If the model checkpoint contains 2D weights and conversion
        mappings areregistered for the model type, weights are loaded directly in
        linearized format.
    2. Post-load conversion: If no conversion mappings exist, the model is loaded
        normally and then linearized via `linearize_moe`.

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

        # model is 3d (or otherwise doesn't have mappings)
        # fall back to post-load conversion
        if not has_linearize_load_mappings(model_type):
            model = original_from_pretrained(*args, **kwargs)
            linearize_moe(model)
            return model

        # prepare to load linearized weights
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

    :param model: model containing MoE layers to linearize
    """
    non_linearized_moes = get_non_linearized_moes(model)

    if len(non_linearized_moes) <= 0:
        logger.warning(
            "Could not find experts to linearize. If your model is an MoE model, "
            "consider registering a linearized class"  # TODO: add docs
        )
        return model

    logger.warning(
        "MoE is being linearized after loading in order to support efficient "
        "calibration of experts. However, this may be inefficient if the model "
        "checkpoint is already linearized (2D -> 3D -> 2D). Consider registering "
        "a load converter for faster load times."  # TODO: add docs
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
    Return all modules which are recognized to be experts layers. A module is recognized
    as an experts layer if it conforms to the `FusedExpertsProtocol` or is registered by
    `LinearExperts2D`.

    :param model: model with modules to check for experts
    :return: list of named modules which are recognized as experts layers
    """
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, FusedExpertsProtocol)
        or LinearExperts2D.get_registration(module.__class__) is not None
    ]
