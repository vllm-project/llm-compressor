from compressed_tensors.offload import dispatch_model
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Load model.
MODEL_ID = "Qwen/Qwen3.5-27B"
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 32 via ptq
#   * skip the visual encoder, lm_head, and linear attention
#   (Gated DeltaNet fused projections are incompatible with microscale formats)

# No need to include mtp layers as they are not loaded
# through Qwen3_5ForConditionalGeneration
recipe = QuantizationModifier(
    targets="Linear",
    scheme="MXFP4A16",
    ignore=[
        "lm_head",
        "re:.*visual.*",
        "re:.*linear_attn.*",
    ],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
messages = [{"role": "user", "content": "Hello my name is"}]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(text=prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================\n\n")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

# MTP layers are excluded from the model through Qwen3_5ForConditionalGeneration
# Save them as-is from the original checkpoint into the quantized output.
save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)
