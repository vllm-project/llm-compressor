import contextlib
from typing import Type

import torch
from compressed_tensors.utils import patch_attr
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    conversion_mapping,
)
from transformers.conversion_mapping import (
    MergeModulelist,
    WeightConverter,
    WeightRenaming,
    WeightTransform,
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
from transformers.monkey_patching import clear_patch_mapping, register_patch_mapping

from .linear_experts import create_linear_experts_2d

# TODO: in the future, we can potentially grep the source code for this
ARCH_TO_EXPERTS_MODULE_CLS = {"deepseek_v4": DeepseekV4Experts}


def extract_moe_merge_operations(
    mapping: list[WeightTransform],
) -> tuple[list[WeightTransform], list[WeightTransform]]:
    merge_mappings, non_merge_mappings = [], []
    for converter in mapping:
        if isinstance(converter, WeightConverter):
            merge_ops, non_merge_ops = [], []
            for op in converter.operations:
                if isinstance(op, MergeModulelist):
                    merge_ops.append(op)
                else:
                    non_merge_ops.append(op)

            merge_mappings.append(
                WeightConverter(
                    source_patterns=converter.source_patterns,
                    target_patterns=converter.target_patterns,
                    operations=merge_ops,
                )
            )

            if len(non_merge_ops) <= 0:
                non_merge_mappings.append(
                    WeightRenaming(
                        source_patterns=converter.source_patterns,
                        target_patterns=converter.target_patterns,
                    )
                )

            else:
                non_merge_mappings.append(
                    WeightConverter(
                        source_patterns=converter.source_patterns,
                        target_patterns=converter.target_patterns,
                        operations=non_merge_ops,
                    )
                )

        else:
            non_merge_mappings.append(converter)

    return merge_mappings, non_merge_mappings


def get_linearization(
    model_type: str,
) -> tuple[type[torch.nn.Module], list[WeightTransform], list[WeightTransform]]:
    # TODO: early exit if not moe model
    experts_cls = ARCH_TO_EXPERTS_MODULE_CLS[model_type]

    mapping: list[WeightTransform] = get_checkpoint_conversion_mapping(model_type)
    merge_mappings, non_merge_mappings = extract_moe_merge_operations(mapping)
    # _2d_moe_mappings = [converter for converter in mapping if _is_2d_converter(converter)]

    if len(merge_mappings) > 0:
        # if checkpoint is in 2d, keep in 2d
        forward_mapping = backwards_mapping = non_merge_mappings

    else:
        # if checkpoint is in 3d, split into 2d, save as 2d
        # TODO: split checkpoint into 2d
        forward_mapping = backwards_mapping = mapping  # if checkpoint is 3d, keep in 3d
        raise ValueError("3d not supported yet")

    # TODO: some sort of validation which checks if any of the mappings have WeightConverters
    # if they do, warn that conversion occurs and that runtime may increase for offloaded models

    return experts_cls, forward_mapping, backwards_mapping


@contextlib.contextmanager
def load_linearized_moe(model_cls: Type[PreTrainedModel] = AutoModelForCausalLM):
    original_from_pretrained = model_cls.from_pretrained

    @classmethod
    def patched(cls, *args, **kwargs):
        config = AutoConfig.from_pretrained(*args, **kwargs)
        model_type = config.model_type
        experts_cls, forward_mapping, backward_mapping = get_linearization(model_type)

        linear_experts_2d_cls = create_linear_experts_2d(experts_cls)
        register_patch_mapping({experts_cls.__name__: linear_experts_2d_cls})
        register_checkpoint_conversion_mapping(
            model_type, forward_mapping, overwrite=True
        )

        model: PreTrainedModel = original_from_pretrained(*args, **kwargs)
        model._conversion_mapping = backward_mapping

        clear_patch_mapping()
        conversion_mapping._checkpoint_conversion_mapping_cache = None
        return model

    with patch_attr(model_cls, "from_pretrained", patched):
        yield
