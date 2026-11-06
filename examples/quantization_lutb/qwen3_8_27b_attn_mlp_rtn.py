"""Round-to-nearest LUT-B quantization of Qwen3.8-27B attention and MLP."""

import os

from compressed_tensors.offload import dispatch_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
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

recipe = QuantizationModifier(targets=TARGETS, scheme="LUTB", ignore=IGNORE)
oneshot(model=model, processor=processor, recipe=recipe)

save_dir = os.environ.get("SAVE_DIR", "Qwen3.8-27B-LUTB-Attn-MLP-RTN")
model.save_pretrained(save_dir, save_compressed=True)
processor.save_pretrained(save_dir)

# LUT-B fake quantization refits codebooks during Transformers forwards. Keep
# the example's optional generation check after the artifact is safely saved.
if os.environ.get("RUN_TRANSFORMERS_SMOKE") == "1":
    print("========== SAMPLE GENERATION ==========")
    dispatch_model(model)
    sample = processor.tokenizer(
        "Explain why the sky appears blue.", return_tensors="pt"
    )
    sample = {key: value.to(model.device) for key, value in sample.items()}
    output = model.generate(
        **sample, max_new_tokens=int(os.environ.get("SMOKE_MAX_NEW_TOKENS", "100"))
    )
    print(processor.tokenizer.decode(output[0], skip_special_tokens=True))
    print("========================================")
