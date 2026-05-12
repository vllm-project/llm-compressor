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
    extract_weight_conversions_for_model,
)
from transformers.integrations.moe import _default_apply_gate
from transformers.modeling_utils import local_torch_dtype
from transformers.monkey_patching import register_patch_mapping
from transformers.conversion_mapping import register_checkpoint_conversion_mapping, WeightConverter


from llmcompressor.utils.dev import skip_weights_initialize

from .linear_experts import LinearExperts

from compressed_tensors.utils import patch_attr


# TODO: in the future, can probably match using regex
ARCH_TO_EXPERTS_MODULE_CLS = {
    "deepseek_v4": "DeepseekV4Experts"
}


def get_linear_conversion_mapping():
    pass


@contextlib.contextmanager
def load_linearized_moe(model_cls: Type[PreTrainedModel] = AutoModelForCausalLM):

    original_from_pretrained = model_cls.from_pretrained

    @classmethod
    def patched(cls, *args, **kwargs):
        config = AutoConfig.from_pretrained(*args, **kwargs)
        model_type = config.model_type

        experts_cls = ARCH_TO_EXPERTS_MODULE_CLS[model_type]
        #forward_mapping, backward_mapping = get_linear_conversion_mapping(model_type)

        register_patch_mapping(
            {experts_cls.__name__: LinearExperts}
        )
        # register_checkpoint_conversion_mapping(
        #     model_type=model_type, mapping=forward_mapping, overwrite=True
        # )

        model: PreTrainedModel = original_from_pretrained(cls, *args, **kwargs)
        #model._conversion_mapping = backward_mapping
        return model

    with patch_attr(model_cls, "from_pretrained", patched):
        yield

