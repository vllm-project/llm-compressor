from abc import ABC
from typing import Callable, Type

import torch
import contextlib
import torch.distributed as dist
import tqdm
from compressed_tensors.distributed import is_distributed
from compressed_tensors.offload import get_cache_init_kwargs, offload_module
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
from transformers.conversion_mapping import (
    extract_weight_conversions_for_model, get_checkpoint_conversion_mapping, WeightTransform
)
from transformers.integrations.moe import _default_apply_gate
from transformers.modeling_utils import local_torch_dtype
from transformers.monkey_patching import register_patch_mapping
from transformers.conversion_mapping import register_checkpoint_conversion_mapping, WeightConverter


from llmcompressor.utils.dev import skip_weights_initialize

from .linear_experts import LinearExperts

from compressed_tensors.utils import patch_attr


ARCH_TO_EXPERTS_MODULE_CLS = {
    "deepseek_v4": "DeepseekV4Experts"
}


def extract_moe_merge_operations(...):
    # TODO


def get_linearization(model_type: str) -> tuple[str, list[WeightTransform], list[WeightTransform]]:
    # TODO: early exit if not moe model
    experts_cls = ARCH_TO_EXPERTS_MODULE_CLS[model_type]

    # get conversion
    def _is_2d_converter(converter: WeightTransform) -> bool:
        return isinstance(converter, WeightConverter) and converter.target_patterns in (
            ["mlp.experts.gate_up_proj"],
            ["mlp.experts.down_proj"],
        )
    
    mapping: list[WeightTransform] = get_checkpoint_conversion_mapping(model_type)
    merge_mappings, without_merge_mappings = extract_moe_merge_operations(mapping)
    #_2d_moe_mappings = [converter for converter in mapping if _is_2d_converter(converter)]
    
    if len(merge_mappings) > 0:
        # if checkpoint is in 2d, keep in 2d
        forward_mapping, backwards_mapping = without_merge_mappings

    else:
        # if checkpoint is in 3d, split into 2d, save as 2d
        # TODO: split checkpoint into 2d
        forward_mapping = backwards_mapping = mapping  # if checkpoint is 3d, keep in 3d
        raise ValueError("3d not supported yet")
    

    # TODO: some sort of validation which checks if any of the mappings have WeightConverters
    # if they do, warn that conversion occurs and that runtime may increase for offloaded models

    return (experts_cls, ), forward_mapping, backwards_mapping


@contextlib.contextmanager
def load_linearized_moe(model_cls: Type[PreTrainedModel] = AutoModelForCausalLM):

    original_from_pretrained = model_cls.from_pretrained

    @classmethod
    def patched(cls, *args, **kwargs):
        config = AutoConfig.from_pretrained(*args, **kwargs)
        model_type = config.model_type
        experts_cls, forward_mapping, backward_mapping = get_linearization(model_type)

        register_patch_mapping(
            {experts_cls.__name__: LinearExperts}
        )
        register_checkpoint_conversion_mapping(
            model_type, forward_mapping, overwrite=True
        )

        model: PreTrainedModel = original_from_pretrained(cls, *args, **kwargs)
        model._conversion_mapping = backward_mapping
        return model

    with patch_attr(model_cls, "from_pretrained", patched):
        yield

