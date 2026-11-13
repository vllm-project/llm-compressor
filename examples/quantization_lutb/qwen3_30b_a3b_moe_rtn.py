"""Round-to-nearest LUT-B quantization of Qwen3-30B-A3B MoE experts."""

import os

from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
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

recipe = QuantizationModifier(targets=TARGETS, scheme="LUTB", ignore=IGNORE)
oneshot(model=model, recipe=recipe)

save_dir = os.environ.get("SAVE_DIR", "Qwen3-30B-A3B-LUTB-MoE-RTN")
model.save_pretrained(save_dir, save_compressed=True)
tokenizer.save_pretrained(save_dir)

# LUT-B fake quantization refits codebooks during Transformers forwards. Keep
# the example's optional generation check after the artifact is safely saved.
if os.environ.get("RUN_TRANSFORMERS_SMOKE") == "1":
    print("========== SAMPLE GENERATION ==========")
    dispatch_model(model)
    sample = tokenizer("Explain why the sky appears blue.", return_tensors="pt")
    sample = {key: value.to(model.device) for key, value in sample.items()}
    output = model.generate(
        **sample, max_new_tokens=int(os.environ.get("SMOKE_MAX_NEW_TOKENS", "100"))
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("========================================")
