"""MTP adaptations for the GlmMoeDsaForCausalLM architecture."""

from copy import deepcopy

from transformers import PretrainedConfig, PreTrainedModel
from transformers.core_model_loading import PrefixChange, WeightRenaming
from transformers.modeling_layers import MtpModel


class GlmMoeDsaMtpModel(MtpModel):
    @staticmethod
    def get_mtp_config(model: PreTrainedModel) -> PretrainedConfig:
        text_config = model.config.get_text_config()
        num_layers = getattr(text_config, "num_nextn_predict_layers", None)
        if num_layers is None:
            num_layers = getattr(text_config, "num_mtp_layers", None)
        if num_layers != 1:
            raise ValueError("GLM-MoE-DSA MTP requires one prediction layer")
        config = deepcopy(text_config)
        config.num_mtp_layers = config.num_hidden_layers = num_layers
        # The draft layer is always MoE with its own DSA indexer, even when
        # the backbone starts with dense layers or ends with a shared indexer.
        config.first_k_dense_replace = 0
        config.mlp_layer_types = ["sparse"]
        config.layer_types = ["indexed_attention"]
        config.indexer_types = ["full"]
        return config

    @staticmethod
    def get_mtp_weight_conversions():
        # Transformers 5.15 supplies the trailing-layer rename before this hook.
        return [
            PrefixChange(prefix_to_remove="model"),
            WeightRenaming(r"^layers\.0\.mtp_block\.enorm\.", "layers.0.enorm."),
            WeightRenaming(r"^layers\.0\.mtp_block\.hnorm\.", "layers.0.hnorm."),
            WeightRenaming(r"^layers\.0\.mtp_block\.eh_proj\.", "layers.0.eh_proj."),
            WeightRenaming(
                r"^layers\.0\.mtp_block\.shared_head\.norm\.",
                "layers.0.post_norm.",
            ),
        ]

    def get_weight_conversions(self, main_model: PreTrainedModel):
        num_hidden_layers = main_model.config.get_text_config().num_hidden_layers
        return [
            WeightRenaming(rf"layers\.{num_hidden_layers}\.", "layers.0.mtp_block."),
            *self.get_mtp_weight_conversions(),
            *main_model._weight_conversions,
        ]
