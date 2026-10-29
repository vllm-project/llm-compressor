"""Delayed-codebook GPTQ LUT-B quantization of Qwen3.8-27B attention and MLP."""

import os

from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq.lutb_delayed import GPTQLutBDelayedModifier
from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen3.8-27B"
TARGETS = [
    # Restrict targets to the language model; the vision tower's 1152x4304 MLP
    # matrices do not align with LUT-B's 8x64 blocks.
    r"re:.*language_model\.layers\..*\.self_attn\..*",
    r"re:.*language_model\.layers\..*\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
    r"re:.*language_model\.layers\..*\.mlp\..*",
]
IGNORE = [
    r"re:.*lm_head$",
    r"re:.*embed_tokens$",
    r"re:.*linear_attn\.in_proj_a$",
    r"re:.*linear_attn\.in_proj_b$",
]

with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

recipe = GPTQLutBDelayedModifier(targets=TARGETS, scheme="LUTB", ignore=IGNORE)
oneshot(
    model=model,
    processor=processor,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=4096,
    num_calibration_samples=512,
)

save_dir = os.environ.get("SAVE_DIR", "Qwen3.8-27B-LUTB-Attn-MLP-GPTQ-Delayed")
model.save_pretrained(save_dir, save_compressed=True)
processor.save_pretrained(save_dir)
