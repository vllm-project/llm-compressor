from transformers import PreTrainedModel
from transformers.conversion_mapping import extract_weight_conversions_for_model
from transformers.core_model_loading import WeightConverter, WeightRenaming


def modify_save_with_linearized_experts_deepseek_v4(model: PreTrainedModel):
    """
    Replace the fused-expert weight converters in the deepseek_v4 conversion
    mapping with per-expert renamings so that checkpoint weights load directly
    into a linearized model (individual ``nn.Linear`` modules per expert)
    instead of being merged into 3-D ``gate_up_proj`` / ``down_proj`` tensors.
    """
    weight_conversions = extract_weight_conversions_for_model(model)
    if weight_conversions is None:
        weight_conversions = []

    new_conversions = [
        conv
        for conv in weight_conversions
        if not _is_fused_experts_converter(conv)
    ]

    new_conversions.extend(
        [
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.w1\.",
                target_patterns=r"layers.\1.mlp.experts.\2.gate_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.w3\.",
                target_patterns=r"layers.\1.mlp.experts.\2.up_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^layers\.(\d+)\.mlp\.experts\.(\d+)\.w2\.",
                target_patterns=r"layers.\1.mlp.experts.\2.down_proj.",
            ),
        ]
    )

    model._weight_conversions = new_conversions


def modify_save_with_linearized_experts_qwen2_moe(model: PreTrainedModel):
    """
    Remove the fused-expert weight converters from the qwen2_moe conversion
    mapping so that checkpoint weights load directly into a linearized model.

    The qwen2_moe checkpoint already uses ``gate_proj`` / ``up_proj`` /
    ``down_proj`` naming, which matches the linearized module names, so no
    replacement renamings are needed — just dropping the ``MergeModulelist`` +
    ``Concatenate`` converters is sufficient.
    """
    weight_conversions = extract_weight_conversions_for_model(model)
    if weight_conversions is None:
        weight_conversions = []

    model._weight_conversions = [
        conv
        for conv in weight_conversions
        if not _is_fused_experts_converter(conv)
    ]


def _is_fused_experts_converter(converter) -> bool:
    return isinstance(converter, WeightConverter) and converter.target_patterns in (
        ["mlp.experts.gate_up_proj"],
        ["mlp.experts.down_proj"],
    )
