import torch
from loguru import logger
from transformers import PreTrainedModel
from transformers.conversion_mapping import (
    _MODEL_TO_CONVERSION_PATTERN,
    get_checkpoint_conversion_mapping,
)
from transformers.core_model_loading import (
    WeightConverter,
    WeightRenaming,
    WeightTransform,
)

__all__ = [
    "has_linearize_load_mappings",
    "get_linearize_load_mappings",
    "set_save_conversion_mapping",
]

ARCH_TO_2D_MAPPINGS = {
    "deepseek_v4": (
        ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        [
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.w1\.",
                target_patterns=r"layers.\1.mlp.experts.\2.gate_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.w2\.",
                target_patterns=r"layers.\1.mlp.experts.\2.down_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.w3\.",
                target_patterns=r"layers.\1.mlp.experts.\2.up_proj.",
            ),
        ],
    ),
    "qwen2_moe": (
        ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        [
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.",
                target_patterns=r"layers.\1.mlp.experts.\2.gate_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.",
                target_patterns=r"layers.\1.mlp.experts.\2.up_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.",
                target_patterns=r"layers.\1.mlp.experts.\2.down_proj.",
            ),
        ],
    ),
}


def get_experts_cls(model_type: str) -> type[torch.nn.Module]:
    if model_type == "deepseek_v4":
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4Experts,
        )

        return DeepseekV4Experts

    elif model_type == "qwen2_moe":
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeExperts

        return Qwen2MoeExperts

    elif model_type == "qwen3_moe":
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

        return Qwen3MoeExperts

    elif model_type == "glm_moe_dsa":
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaExperts,
        )

        return GlmMoeDsaExperts

    else:
        raise ValueError()


def has_linearize_load_mappings(model_type: str) -> bool:
    model_type = _MODEL_TO_CONVERSION_PATTERN.get(model_type, model_type)
    return model_type in ARCH_TO_2D_MAPPINGS


def get_linearize_load_mappings(
    model_type: str,
) -> tuple[type[torch.nn.Module], list[WeightTransform], list[WeightTransform]]:
    """ """
    experts_cls = get_experts_cls(model_type)
    model_type = _MODEL_TO_CONVERSION_PATTERN.get(model_type, model_type)

    mapping: list[WeightTransform] = get_checkpoint_conversion_mapping(model_type)
    remove_targets, new_mappings = ARCH_TO_2D_MAPPINGS[model_type]

    # forwards has conversion mappings
    # backwards has no mappings (stay 2d)
    save_mappings = [
        converter
        for converter in mapping
        if not any(target in remove_targets for target in converter.target_patterns)
    ]
    load_mappings = save_mappings + new_mappings

    # validate that no transforms occur during loading/saving
    for converter in load_mappings:
        if isinstance(converter, WeightConverter):
            logger.warning(
                "Linearized model performs a weight conversion during loading. This "
                f"may lead to longer load times\n{converter}"
            )
    for converter in save_mappings:
        if isinstance(converter, WeightConverter):
            logger.warning(
                "Linearized model performs a weight conversion during saving. This "
                f"may lead to longer save times\n{converter}"
            )

    return experts_cls, load_mappings, save_mappings


def set_save_conversion_mapping(
    model: PreTrainedModel, save_mappings: list[WeightTransform]
):
    """
    Set the conversion mappings used when saving the model. The inverse of these
    mappings will be applied to the model during saving via
    `transformers.core_model_loading.py::revert_weight_conversion`.

    :param model: model to override conversion mapping of
    :param save_mappings: mappings to override with
    """
    model._weight_conversions = save_mappings
