"""GPTQ LUT-B quantization of Qwen3-30B-A3B MoE experts."""

import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQLutBModifier
from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen3-30B-A3B"
TARGETS = r"re:.*mlp\.experts\..*\.(gate|up|down)_proj$"
IGNORE = [
    r"re:.*lm_head$",
    r"re:.*mlp\.gate$",
    r"re:.*shared_expert_gate.*",
]

with load_context():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = GPTQLutBModifier(targets=TARGETS, scheme="LUTB", ignore=IGNORE)
oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
    moe_calibrate_all_experts=True,
)

save_dir = os.environ.get("SAVE_DIR", "Qwen3-30B-A3B-LUTB-MoE-GPTQ")
model.save_pretrained(save_dir, save_compressed=True)
tokenizer.save_pretrained(save_dir)
