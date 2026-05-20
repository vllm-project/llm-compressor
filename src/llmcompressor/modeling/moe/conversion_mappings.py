import torch
from loguru import logger
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
)
from transformers.core_model_loading import (
    WeightConverter,
    WeightRenaming,
    WeightTransform,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

# TODO: in the future, we can potentially grep the source code for this
ARCH_TO_EXPERTS_MODULE_CLS = {"deepseek_v4": DeepseekV4Experts}

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
    )
}


def _has_2d_mappings(model_type: str) -> bool:
    return model_type in ARCH_TO_2D_MAPPINGS


def _get_2d_mappings(
    model_type: str,
) -> tuple[type[torch.nn.Module], list[WeightTransform], list[WeightTransform]]:
    # TODO: early exit if not moe model
    experts_cls = ARCH_TO_EXPERTS_MODULE_CLS[model_type]

    mapping: list[WeightTransform] = get_checkpoint_conversion_mapping(model_type)
    remove_targets, new_mappings = ARCH_TO_2D_MAPPINGS[model_type]

    # forwards has conversion mappings
    # backwards has no mappings (stay 2d)
    backward_mappings = [
        converter
        for converter in mapping
        if not any(target in remove_targets for target in converter.target_patterns)
    ]
    forward_mappings = backward_mappings + new_mappings

    # validate that no transforms occur during loading/saving
    for converter in forward_mappings:
        if isinstance(converter, WeightConverter):
            logger.warning(
                "Linearized model performs a weight conversion during loading. This "
                f"may lead to longer load times\n{converter}"
            )
    for converter in backward_mappings:
        if isinstance(converter, WeightConverter):
            logger.warning(
                "Linearized model performs a weight conversion during saving. This "
                f"may lead to longer save times\n{converter}"
            )

    return experts_cls, forward_mappings, backward_mappings
