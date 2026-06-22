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
from transformers.models.deepseek_ocr2.modeling_deepseek_ocr2 import (
    DeepseekOcr2TextExperts,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
from transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe import (
    Ernie4_5_VLMoeMoeExperts,
)
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
from transformers.models.hy_v3.modeling_hy_v3 import HYV3Experts
from transformers.models.jamba.modeling_jamba import JambaExperts
from transformers.models.laguna.modeling_laguna import LagunaExperts
from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeExperts
from transformers.models.minimax.modeling_minimax import MiniMaxExperts
from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2Experts
from transformers.models.mixtral.modeling_mixtral import MixtralExperts
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts
from transformers.models.openai_privacy_filter.modeling_openai_privacy_filter import (
    OpenAIPrivacyFilterExperts,
)
from transformers.models.phimoe.modeling_phimoe import PhimoeExperts
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeExperts
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

__all__ = [
    "has_linearize_load_mappings",
    "get_linearize_load_mappings",
    "set_save_conversion_mapping",
]

ARCH_TO_EXPERTS_MODULE_CLS = {
    "deepseek_ocr2": DeepseekOcr2TextExperts,
    "deepseek_v4": DeepseekV4Experts,
    "ernie4_5_vl_moe": Ernie4_5_VLMoeMoeExperts,
    "gemma4": Gemma4TextExperts,
    "gpt_oss": GptOssExperts,
    "hy_v3": HYV3Experts,
    "jamba": JambaExperts,
    "laguna": LagunaExperts,
    "lfm2_moe": Lfm2MoeExperts,
    "minimax": MiniMaxExperts,
    "minimax_m2": MiniMaxM2Experts,
    "mixtral": MixtralExperts,
    "nemotron_h": NemotronHExperts,
    "openai_privacy_filter": OpenAIPrivacyFilterExperts,
    "phimoe": PhimoeExperts,
    "qwen2_moe": Qwen2MoeExperts,
    "qwen3_5_moe": Qwen3_5MoeExperts,
    "qwen3_moe": Qwen3MoeExperts,
    "qwen3_vl_moe": Qwen3VLMoeTextExperts,
}

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
    "phimoe": (
        [".experts.gate_up_proj", ".experts.down_proj"],
        [
            WeightRenaming(
                source_patterns=r"^model.layers\.(\d+)\.mlp\.experts\.(\d+)\.w1\.",
                target_patterns=r"model.layers.\1.mlp.experts.\2.gate_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^model.layers\.(\d+)\.mlp\.experts\.(\d+)\.w3\.",
                target_patterns=r"model.layers.\1.mlp.experts.\2.up_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^model.layers\.(\d+)\.mlp\.experts\.(\d+)\.w2\.",
                target_patterns=r"model.layers.\1.mlp.experts.\2.down_proj.",
            ),
        ],
    ),
    "qwen2_moe": (
        ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        [
            WeightRenaming(
                source_patterns=r"^model.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.",
                target_patterns=r"model.layers.\1.mlp.experts.\2.gate_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^model.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.",
                target_patterns=r"model.layers.\1.mlp.experts.\2.up_proj.",
            ),
            WeightRenaming(
                source_patterns=r"^model.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.",
                target_patterns=r"model.layers.\1.mlp.experts.\2.down_proj.",
            ),
        ],
    ),
}


def has_linearize_load_mappings(model_type: str) -> bool:
    model_type = _MODEL_TO_CONVERSION_PATTERN.get(model_type, model_type)
    return model_type in ARCH_TO_2D_MAPPINGS


def get_linearize_load_mappings(
    model_type: str,
) -> tuple[type[torch.nn.Module], list[WeightTransform], list[WeightTransform]]:
    """ """
    experts_cls = ARCH_TO_EXPERTS_MODULE_CLS[model_type]
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
