"""MTP adaptations for the Qwen3_5ForConditionalGeneration architecture."""

from copy import deepcopy

from transformers import PretrainedConfig, PreTrainedModel
from transformers.core_model_loading import PrefixChange, WeightRenaming
from transformers.modeling_layers import MtpModel


class Qwen3_5MtpModel(MtpModel):
    @staticmethod
    def get_mtp_config(model: PreTrainedModel) -> PretrainedConfig:
        # Qwen's hybrid backbone drafts with one full-attention layer and shares
        # its embedding/head with the backbone.
        text_config = model.config.get_text_config()
        num_layers = getattr(text_config, "mtp_num_hidden_layers", None)
        if num_layers != 1 or getattr(
            text_config, "mtp_use_dedicated_embeddings", False
        ):
            raise ValueError("Qwen3.5 MTP requires one layer and shared embeddings")
        config = deepcopy(text_config)
        config.num_mtp_layers = config.num_hidden_layers = num_layers
        config.layer_types = ["full_attention"] * num_layers
        return config

    @staticmethod
    def get_mtp_weight_conversions():
        # Explicit, reversible mappings also support Transformers versions without
        # this layout; its generic negative-lookahead block mapping can't round-trip.
        return [
            PrefixChange(prefix_to_remove="mtp"),
            WeightRenaming(r"^layers\.0\.", "layers.0.mtp_block."),
            WeightRenaming(r"^fc\.", "layers.0.eh_proj."),
            WeightRenaming(r"^pre_fc_norm_embedding\.", "layers.0.enorm."),
            WeightRenaming(r"^pre_fc_norm_hidden\.", "layers.0.hnorm."),
            WeightRenaming(r"^norm\.", "layers.0.post_norm."),
        ]

    def get_weight_conversions(self, main_model: PreTrainedModel):
        return [*self.get_mtp_weight_conversions(), *main_model._weight_conversions]
