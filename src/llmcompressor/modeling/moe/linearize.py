import contextlib
from functools import wraps
from typing import Type

import torch
from compressed_tensors.utils import patch_attr
from loguru import logger
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    conversion_mapping,
)
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.core_model_loading import (
    WeightConverter,
    WeightRenaming,
    WeightTransform,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
from transformers.monkey_patching import clear_patch_mapping, register_patch_mapping

from llmcompressor.modeling.moe.helpers import FusedExpertsProtocol
from .conversion_mappings import _has_2d_mappings, _get_2d_mappings

from .linear_experts import LinearExperts2D


@contextlib.contextmanager
def load_quantizable_moe(model_cls: Type[PreTrainedModel] = AutoModelForCausalLM):
    original_from_pretrained = model_cls.from_pretrained

    @classmethod
    @wraps(original_from_pretrained)
    def patched(cls, *args, **kwargs):
        config = AutoConfig.from_pretrained(*args, **kwargs)
        model_type = config.model_type

        # model is 3d (or otherwise doesn't have mappings)
        # fall back to post-load conversion
        if not _has_2d_mappings(model_type):
            model: PreTrainedModel = original_from_pretrained(*args, **kwargs)
            return linearize_moe(model)

        # prepare to load linearized weights
        experts_cls, forward_mapping, backward_mapping = _get_2d_mappings(model_type)
        linear_experts_2d_cls = LinearExperts2D.create_linear_experts_cls(experts_cls)
        register_patch_mapping({experts_cls.__name__: linear_experts_2d_cls})
        register_checkpoint_conversion_mapping(
            model_type, forward_mapping, overwrite=True
        )

        # load model
        model: PreTrainedModel = original_from_pretrained(*args, **kwargs)

        # prepare for saving to be called later
        clear_patch_mapping()
        model._conversion_mapping = backward_mapping
        conversion_mapping._checkpoint_conversion_mapping_cache = None

        return model

    with patch_attr(model_cls, "from_pretrained", patched):
        yield


def linearize_moe(model: PreTrainedModel) -> PreTrainedModel:
    non_linearized_moes = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, FusedExpertsProtocol)
    }

    if len(non_linearized_moes) <= 0:
        logger.warning("TODO could not find experts to replace")
        return model

    logger.warning("TODO")
    original_experts_cls = next(non_linearized_moes.values())
    linear_experts_cls = LinearExperts2D.create_linear_experts_cls(original_experts_cls)
    for name, module in non_linearized_moes.items():
        linear_moe = linear_experts_cls.from_experts_module(module, model.config)
        model.set_submodule(name, linear_moe)
