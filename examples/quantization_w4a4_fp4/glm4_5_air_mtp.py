"""Calibrate GLM-4.5-Air and its MTP layer together to NVFP4."""

import os
from types import MethodType

import torch
from compressed_tensors.offload import init_dist, set_onload_device
from compressed_tensors.utils import patch_attr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_layers import MtpModel
from transformers.monkey_patching import clear_patch_mapping, register_patch_mapping

from llmcompressor import oneshot
from llmcompressor.modeling.moe.conversion_mappings import get_linearize_load_mappings
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "zai-org/GLM-4.5-Air"
SAVE_DIR = os.environ.get("MTP_OUTPUT_DIR", "GLM-4.5-Air-NVFP4-MTP")
OFFLOAD_DIR = os.environ.get("MTP_OFFLOAD_DIR", "offload_folder")

init_dist()
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        max_memory={"cpu": "500GiB"},
        offload_folder=OFFLOAD_DIR,
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
experts_cls, _, _ = get_linearize_load_mappings(model.config.model_type)
register_patch_mapping(
    {experts_cls.__name__: LinearExperts2D.get_linear_experts_cls(experts_cls)}
)
try:
    model.mtp = MtpModel.from_pretrained(model)
finally:
    clear_patch_mapping()
if len(model.mtp.layers) != 1:
    raise ValueError("This example expects one MTP layer")
set_onload_device(model, "cuda")

backbone_forward = model.forward


def forward_with_mtp(self, input_ids, attention_mask=None, position_ids=None, **kw):
    set_onload_device(self, "cuda")
    outputs = backbone_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        use_cache=False,
        **kw,
    )
    if input_ids.shape[1] < 2:
        raise ValueError("MTP calibration requires at least two tokens")
    if position_ids is None:
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
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


recipe = QuantizationModifier(
    targets="Linear", scheme="NVFP4", ignore=["lm_head", r"re:.*\.eh_proj$"]
)
with patch_attr(model, "forward", MethodType(forward_with_mtp, model)):
    oneshot(
        model=model,
        processor=tokenizer,
        dataset="perfectblend",
        splits="train[:512]",
        recipe=recipe,
        pipeline="basic",
        max_seq_length=512,
        num_calibration_samples=20,
    )

model.save_pretrained(SAVE_DIR, mtp_quant_scheme="NVFP4")
tokenizer.save_pretrained(SAVE_DIR)
torch.distributed.destroy_process_group()
