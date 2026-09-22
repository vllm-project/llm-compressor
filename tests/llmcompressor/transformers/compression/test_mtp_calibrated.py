import json
from types import MethodType
from unittest.mock import patch

import torch
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.utils.data import DataLoader
from transformers import Glm4MoeConfig, Glm4MoeForCausalLM, PreTrainedTokenizerFast
from transformers.modeling_layers import MtpModel

from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modifiers.quantization import QuantizationModifier


def test_one_calibration_run_saves_nvfp4_backbone_and_mtp(tmp_path):
    config = Glm4MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=16,
        n_routed_experts=2,
        num_experts_per_tok=1,
        first_k_dense_replace=1,
        num_nextn_predict_layers=1,
        max_position_embeddings=128,
    )
    model = Glm4MoeForCausalLM(config).eval()
    source = tmp_path / "source"
    model.save_pretrained(source)
    model.config.name_or_path = str(source)
    model.config.dtype = torch.float32
    model.mtp = MtpModel(model, 1).eval()
    linearize_moe(model)

    backbone_forward = model.forward

    def forward_with_mtp(self, input_ids, attention_mask=None, position_ids=None, **kw):
        outputs = backbone_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
            **kw,
        )
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        positions = position_ids[:, 1:]
        embeddings = self.mtp.embed_tokens(input_ids[:, 1:])
        rotary = self.mtp.rotary_emb(embeddings, position_ids=positions)
        masks = self.mtp.create_masks_for_mtp_layer(0, embeddings, None, positions)
        self.mtp.layers[0](
            embeddings,
            outputs.hidden_states[-1][:, :-1],
            position_embeddings=rotary,
            position_ids=positions,
            past_key_values=None,
            **masks,
        )
        return outputs

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    data = DataLoader(
        [{"input_ids": torch.randint(0, 128, (8,))} for _ in range(2)],
        batch_size=1,
    )
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=["lm_head", r"re:.*\.eh_proj$"],
    )
    with patch.object(model, "forward", MethodType(forward_with_mtp, model)):
        oneshot(
            model=model,
            processor=tokenizer,
            dataset=data,
            recipe=recipe,
            pipeline="basic",
            num_calibration_samples=2,
        )

    backbone = model.model.layers[0].self_attn.q_proj
    backbone_expert = model.model.layers[1].mlp.experts[0].up_proj
    mtp = model.mtp.layers[0].mtp_block.self_attn.q_proj
    mtp_expert = model.mtp.layers[0].mtp_block.mlp.experts[0].up_proj
    assert backbone.quantization_status == QuantizationStatus.FROZEN
    assert backbone_expert.quantization_status == QuantizationStatus.FROZEN
    assert mtp.quantization_status == QuantizationStatus.FROZEN
    assert mtp_expert.quantization_status == QuantizationStatus.FROZEN

    destination = tmp_path / "destination"
    model.save_pretrained(destination, mtp_quant_scheme="NVFP4")

    weights = get_weight_mappings(destination)
    assert "model.layers.0.self_attn.q_proj.weight_packed" in weights
    assert "model.layers.1.mlp.experts.0.up_proj.weight_packed" in weights
    assert "model.layers.2.self_attn.q_proj.weight_packed" in weights
    assert "model.layers.2.mlp.experts.0.up_proj.weight_packed" in weights
    assert "model.layers.0.self_attn.q_proj.weight" not in weights
    assert "model.layers.2.self_attn.q_proj.weight" not in weights
    assert "model.layers.2.eh_proj.weight" in weights
    assert "model.layers.2.shared_head.norm.weight" in weights
    assert "model.layers.2.post_norm.weight" not in weights
    assert not any(name.startswith("mtp.") for name in weights)
    scale_name = "model.layers.2.self_attn.q_proj.input_global_scale"
    with safe_open(weights[scale_name], framework="pt") as shard:
        scale = shard.get_tensor(scale_name)
    assert torch.isfinite(scale).all() and torch.all(scale > 0)
    with open(destination / "config.json", encoding="utf-8") as file:
        saved_config = json.load(file)
    assert "model.layers.2.eh_proj" in saved_config["quantization_config"]["ignore"]
    assert model.mtp.layers[0].mtp_block.self_attn.q_proj is mtp
