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
    register_checkpoint_conversion_mapping,
)
from transformers.monkey_patching import clear_patch_mapping, register_patch_mapping

from llmcompressor.modeling.moe.helpers import FusedExpertsProtocol

from .conversion_mappings import _get_2d_mappings, _has_2d_mappings
from .linear_experts import LinearExperts2D

from llmcompressor.utils.dev import skip_weights_initialize


@contextlib.contextmanager
def load_quantizable_moe(model_cls: Type[PreTrainedModel] = AutoModelForCausalLM):
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
        if not _has_2d_mappings(model_type):
            model = original_from_pretrained(*args, **kwargs)
            linearize_moe(model)
            return model

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
    non_linearized_moes = {
        name: module
        for name, module in model.named_modules()
        #if isinstance(module, FusedExpertsProtocol)
        if module.__class__.__name__ == "GraniteMoeParallelExperts"
    }

    if len(non_linearized_moes) <= 0:
        logger.warning("TODO could not find experts to replace")
        return model

    # logger.warning("TODO")
    # original_experts_cls = next(iter(non_linearized_moes.values())).__class__
    # linear_experts_cls = LinearExperts2D.create_linear_experts_cls(original_experts_cls)
    # for name, module in non_linearized_moes.items():
    #     linear_moe = linear_experts_cls.from_experts_module(module, model.config)
    #     model.set_submodule(name, linear_moe)

    for name, module in non_linearized_moes.items():
        linearized = GraniteMoeLinearExperts.from_3d(module)
        model.set_submodule(name, linearized)

    return model
        


def requires_linearize_moe(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, FusedExpertsProtocol):
            return True

    return False


class GraniteMoeLinearExperts(torch.nn.ModuleList):
    @classmethod
    @torch.no_grad()
    def from_3d(cls, original: "GraniteMoeParallelExperts"):
        with skip_weights_initialize():
            self = cls(original.num_experts, original.input_size, original.output_size)

        for i in range(original.num_experts):
            self[i].weight.copy_(original.weight[i])

        return self

    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None:
        """
        Initialize the GraniteMoeParallelExperts module.
        The experts weights are stored in [num_experts, output_size, input_size] format. Such that it's compatible with
        many MoE libraries, such as [Megablock](https://github.com/databricks/megablocks) and
        [ScatterMoE](https://github.com/shawntan/scattermoe), as well as the
        [MoE kernel](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py)
        used in vllm.

        Args:
            num_experts (int):
                Number of experts.
            input_size (int):
                Size of the input.
            output_size (int):
                Size of the output.
        """
        super().__init__([
            torch.nn.Linear(input_size, output_size, bias=False, dtype=torch.bfloat16) for _ in range(num_experts)
        ])
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, inputs, expert_size):
        """
        Forward pass of the GraniteMoeParallelExperts module.

        Args:
            inputs (Tensor):
                Input tensor.
            expert_size:
                Expert size information.

        Returns:
            Tensor: Output tensor.
        """
        input_list = inputs.split(expert_size, dim=0)  # [num_experts, num_tokens_selected, D]
        output_list = []

        for i in range(self.num_experts):
            expert_out = self[i](input_list[i].unsqueeze(0))[0]
            output_list.append(expert_out)
        results = torch.cat(output_list, dim=0)
        return results