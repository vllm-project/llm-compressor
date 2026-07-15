import torch
from loguru import logger
from transformers import PreTrainedModel
from transformers.conversion_mapping import (
    _MODEL_TO_CONVERSION_PATTERN,
    get_checkpoint_conversion_mapping,
)
from transformers.core_model_loading import (
    Concatenate,
    MergeModulelist,
    Transpose,
    WeightConverter,
    WeightRenaming,
    WeightTransform,
)

from .helpers import import_or_none

__all__ = [
    "get_3d_packed_backwards_mappings",
    "get_linearize_load_mappings",
    "has_3d_packed_save_mappings",
    "has_linearize_load_mappings",
    "maybe_set_3d_packed_save_mappings",
    "set_save_conversion_mapping",
]

ARCH_TO_IMPORT_PATHS: dict[str, tuple[str | list[str], str | list[str]]] = {
    "afmoe": (
        "transformers.models.afmoe.configuration_afmoe.AfmoeConfig",
        "transformers.models.afmoe.modeling_afmoe.AfmoeExperts",
    ),
    "cohere2_moe": (
        "transformers.models.cohere2_moe.configuration_cohere2_moe.Cohere2MoeConfig",
        "transformers.models.cohere2_moe.modeling_cohere2_moe.Cohere2MoeExperts",
    ),
    "deepseek_ocr2": (
        "transformers.models.deepseek_ocr2.configuration_deepseek_ocr2.DeepseekOcr2TextConfig",
        "transformers.models.deepseek_ocr2.modeling_deepseek_ocr2.DeepseekOcr2TextExperts",
    ),
    "deepseek_v3": (
        "transformers.models.deepseek_v3.configuration_deepseek_v3.DeepseekV3Config",
        [
            "transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3NaiveMoe",
            "transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3Experts",
        ],
    ),
    "deepseek_v4": (
        "transformers.models.deepseek_v4.configuration_deepseek_v4.DeepseekV4Config",
        "transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4Experts",
    ),
    "exaone_moe": (
        "transformers.models.exaone_moe.configuration_exaone_moe.ExaoneMoeConfig",
        "transformers.models.exaone_moe.modeling_exaone_moe.ExaoneMoeExperts",
    ),
    "flex_olmo": (
        "transformers.models.flex_olmo.configuration_flex_olmo.FlexOlmoConfig",
        "transformers.models.flex_olmo.modeling_flex_olmo.FlexOlmoExperts",
    ),
    "gemma4": (
        "transformers.models.gemma4.configuration_gemma4.Gemma4TextConfig",
        "transformers.models.gemma4.modeling_gemma4.Gemma4TextExperts",
    ),
    "glm4_moe": (
        "transformers.models.glm4_moe.configuration_glm4_moe.Glm4MoeConfig",
        [
            "transformers.models.glm4_moe.modeling_glm4_moe.Glm4MoeNaiveMoe",
            "transformers.models.glm4_moe.modeling_glm4_moe.Glm4MoeExperts",
        ],
    ),
    "glm4_moe_lite": (
        "transformers.models.glm4_moe_lite.configuration_glm4_moe_lite.Glm4MoeLiteConfig",
        [
            "transformers.models.glm4_moe_lite.modeling_glm4_moe_lite.Glm4MoeLiteNaiveMoe",
            "transformers.models.glm4_moe_lite.modeling_glm4_moe_lite.Glm4MoeLiteExperts",
        ],
    ),
    "glm_moe_dsa": (
        "transformers.models.glm_moe_dsa.configuration_glm_moe_dsa.GlmMoeDsaConfig",
        [
            "transformers.models.glm_moe_dsa.modeling_glm_moe_dsa.GlmMoeDsaNaiveMoe",
            "transformers.models.glm_moe_dsa.modeling_glm_moe_dsa.GlmMoeDsaExperts",
        ],
    ),
    "gpt_oss": (
        "transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig",
        "transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts",
    ),
    "granitemoe": (
        "transformers.models.granitemoe.configuration_granitemoe.GraniteMoeConfig",
        [
            "transformers.models.granitemoe.modeling_granitemoe.GraniteMoeParallelExperts",
            "transformers.models.granitemoe.modeling_granitemoe.GraniteMoeExperts",
        ],
    ),
    "hy_v3": (
        "transformers.models.hy_v3.configuration_hy_v3.HYV3Config",
        "transformers.models.hy_v3.modeling_hy_v3.HYV3Experts",
    ),
    "jamba": (
        "transformers.models.jamba.configuration_jamba.JambaConfig",
        "transformers.models.jamba.modeling_jamba.JambaExperts",
    ),
    "laguna": (
        "transformers.models.laguna.configuration_laguna.LagunaConfig",
        "transformers.models.laguna.modeling_laguna.LagunaExperts",
    ),
    "lfm2_moe": (
        "transformers.models.lfm2_moe.configuration_lfm2_moe.Lfm2MoeConfig",
        "transformers.models.lfm2_moe.modeling_lfm2_moe.Lfm2MoeExperts",
    ),
    "llama4": (
        "transformers.models.llama4.configuration_llama4.Llama4TextConfig",
        "transformers.models.llama4.modeling_llama4.Llama4TextExperts",
    ),
    "mellum": (
        "transformers.models.mellum.configuration_mellum.MellumConfig",
        "transformers.models.mellum.modeling_mellum.MellumExperts",
    ),
    "minimax": (
        "transformers.models.minimax.configuration_minimax.MiniMaxConfig",
        "transformers.models.minimax.modeling_minimax.MiniMaxExperts",
    ),
    "minimax_m2": (
        "transformers.models.minimax_m2.configuration_minimax_m2.MiniMaxM2Config",
        "transformers.models.minimax_m2.modeling_minimax_m2.MiniMaxM2Experts",
    ),
    "mixtral": (
        "transformers.models.mixtral.configuration_mixtral.MixtralConfig",
        "transformers.models.mixtral.modeling_mixtral.MixtralExperts",
    ),
    "nemotron_h": (
        "transformers.models.nemotron_h.configuration_nemotron_h.NemotronHConfig",
        "transformers.models.nemotron_h.modeling_nemotron_h.NemotronHExperts",
    ),
    "olmoe": (
        "transformers.models.olmoe.configuration_olmoe.OlmoeConfig",
        "transformers.models.olmoe.modeling_olmoe.OlmoeExperts",
    ),
    "openai_privacy_filter": (
        "transformers.models.openai_privacy_filter.configuration_openai_privacy_filter.OpenAIPrivacyFilterConfig",
        "transformers.models.openai_privacy_filter.modeling_openai_privacy_filter.OpenAIPrivacyFilterExperts",
    ),
    "phimoe": (
        "transformers.models.phimoe.configuration_phimoe.PhimoeConfig",
        "transformers.models.phimoe.modeling_phimoe.PhimoeExperts",
    ),
    "qwen2_moe": (
        "transformers.models.qwen2_moe.configuration_qwen2_moe.Qwen2MoeConfig",
        "transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeExperts",
    ),
    "qwen3_5_moe": (
        "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe.Qwen3_5MoeTextConfig",
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeExperts",
    ),
    "qwen3_moe": (
        "transformers.models.qwen3_moe.configuration_qwen3_moe.Qwen3MoeConfig",
        "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts",
    ),
    "qwen3_next": (
        "transformers.models.qwen3_next.configuration_qwen3_next.Qwen3NextConfig",
        "transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextExperts",
    ),
    "qwen3_vl_moe": (
        "transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe.Qwen3VLMoeTextConfig",
        "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts",
    ),
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


# Architectures whose native checkpoint stores experts as fused 3D tensors.
# After linearization, save-time backwards mappings repack 2D experts (and
# qparams) into that packed layout so Transformers / vLLM can reload the
# checkpoint. See https://github.com/vllm-project/llm-compressor/issues/2699
#
# Pack converters are defined in the SAVE direction (2D model -> 3D disk).
# `get_3d_packed_backwards_mappings` stores their reverse as
# `_weight_conversions` so `revert_weight_conversion` packs on save.
#
# Use `$` anchors on weight / qparam suffixes so `weight_scale` does not
# match the `weight` pattern.
def _qwen3_vl_moe_pack_converters() -> list[WeightConverter]:
    """2D linearized experts -> native qwen3_vl_moe 3D packed checkpoint."""
    # HF fused layout is [E, 2I, H] / [E, H, I]; disk stores the transpose of dims 1/2.
    return [
        WeightConverter(
            source_patterns=[
                "mlp.experts.*.gate_proj.weight$",
                "mlp.experts.*.up_proj.weight$",
            ],
            target_patterns="mlp.experts.gate_up_proj",
            operations=[
                MergeModulelist(dim=0),
                Concatenate(dim=1),
                Transpose(1, 2, check_dims=False),
            ],
        ),
        WeightConverter(
            source_patterns="mlp.experts.*.down_proj.weight$",
            target_patterns="mlp.experts.down_proj",
            operations=[MergeModulelist(dim=0), Transpose(1, 2, check_dims=False)],
        ),
        # Channel / group scales: stack experts then concat gate+up along last dim.
        WeightConverter(
            source_patterns=[
                "mlp.experts.*.gate_proj.weight_scale$",
                "mlp.experts.*.up_proj.weight_scale$",
            ],
            target_patterns="mlp.experts.gate_up_proj_scale",
            operations=[MergeModulelist(dim=0), Concatenate(dim=-1)],
        ),
        WeightConverter(
            source_patterns="mlp.experts.*.down_proj.weight_scale$",
            target_patterns="mlp.experts.down_proj_scale",
            operations=[MergeModulelist(dim=0)],
        ),
        WeightConverter(
            source_patterns=[
                "mlp.experts.*.gate_proj.weight_zero_point$",
                "mlp.experts.*.up_proj.weight_zero_point$",
            ],
            target_patterns="mlp.experts.gate_up_proj_zero_point",
            operations=[MergeModulelist(dim=0), Concatenate(dim=-1)],
        ),
        WeightConverter(
            source_patterns="mlp.experts.*.down_proj.weight_zero_point$",
            target_patterns="mlp.experts.down_proj_zero_point",
            operations=[MergeModulelist(dim=0)],
        ),
    ]


# model_type -> pack converters (SAVE direction). Remapped types resolve via
# `_MODEL_TO_CONVERSION_PATTERN` in the helpers below.
_QWEN3_VL_MOE_PACK = _qwen3_vl_moe_pack_converters()
ARCH_TO_3D_PACK_CONVERTERS: dict[str, list[WeightConverter]] = {
    "qwen3_vl_moe": _QWEN3_VL_MOE_PACK,
    # text_config.model_type for Qwen3-VL-MoE multimodal wrappers
    "qwen3_vl_moe_text": _QWEN3_VL_MOE_PACK,
}


def has_linearize_load_mappings(model_type: str) -> bool:
    remapped_type = _MODEL_TO_CONVERSION_PATTERN.get(model_type, model_type)
    return model_type in ARCH_TO_IMPORT_PATHS and remapped_type in ARCH_TO_2D_MAPPINGS


def get_linearize_load_mappings(
    model_type: str,
) -> tuple[type[torch.nn.Module], list[WeightTransform], list[WeightTransform]]:
    """ """
    _config_paths, expert_paths = ARCH_TO_IMPORT_PATHS[model_type]
    experts_cls = import_or_none(expert_paths)

    mapping: list[WeightTransform] = get_checkpoint_conversion_mapping(model_type)
    model_type = _MODEL_TO_CONVERSION_PATTERN.get(model_type, model_type)
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


def _resolve_3d_pack_key(model_type: str) -> str | None:
    """Return the ARCH_TO_3D_PACK_CONVERTERS key for a model_type, if any."""
    if model_type in ARCH_TO_3D_PACK_CONVERTERS:
        return model_type
    remapped = _MODEL_TO_CONVERSION_PATTERN.get(model_type, model_type)
    if remapped in ARCH_TO_3D_PACK_CONVERTERS:
        return remapped
    return None


def has_3d_packed_save_mappings(model_type: str) -> bool:
    """Whether this architecture should repack linearized experts to 3D on save."""
    return _resolve_3d_pack_key(model_type) is not None


def get_3d_packed_backwards_mappings(model_type: str) -> list[WeightTransform]:
    """
    Return load-direction mappings whose reverse packs 2D linearized experts
    into the architecture's native 3D checkpoint layout on save.

    These should be assigned to ``model._weight_conversions`` (see
    :func:`set_save_conversion_mapping`).
    """
    key = _resolve_3d_pack_key(model_type)
    if key is None:
        raise KeyError(f"No 3D packed save mappings registered for {model_type}")

    pack_converters = ARCH_TO_3D_PACK_CONVERTERS[key]
    # reverse_transform so revert_weight_conversion applies the pack on save
    backwards = [converter.reverse_transform() for converter in pack_converters]
    logger.info(
        f"Using 3D packed backwards mappings for '{model_type}' "
        f"({len(backwards)} converters); save will repack linearized experts"
    )
    return backwards


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


def maybe_set_3d_packed_save_mappings(model: PreTrainedModel) -> bool:
    """
    If ``model.config.model_type`` (or its text config) needs 3D packed saves,
    set ``_weight_conversions`` accordingly.

    :return: True if mappings were installed
    """
    config = getattr(model, "config", None)
    if config is None:
        return False

    def _get_model_type(cfg) -> str | None:
        # HF configs (or nested text_config) can be dict-like during some
        # load/save paths; getattr on a dict would silently miss model_type.
        if isinstance(cfg, dict):
            return cfg.get("model_type")
        return getattr(cfg, "model_type", None)

    candidates = [_get_model_type(config)]
    text_config = getattr(config, "text_config", None)
    if text_config is None and isinstance(config, dict):
        text_config = config.get("text_config")

    if text_config is None:
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            text_config = get_text_config()

    if text_config is not None:
        candidates.append(_get_model_type(text_config))

    for model_type in candidates:
        if model_type and has_3d_packed_save_mappings(model_type):
            set_save_conversion_mapping(
                model, get_3d_packed_backwards_mappings(model_type)
            )
            return True
    return False
