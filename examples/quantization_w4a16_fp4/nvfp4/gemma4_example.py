# Gemma 4 requires transformers >= 5.5.0 (model_type: gemma4).
# If your llmcompressor pins an older version, install with:
#   pip install llmcompressor
#   pip install transformers>=5.5

from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Load model.
MODEL_ID = "google/gemma-4-E4B-it"
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 16 via ptq
#   * skip the vision encoder, audio encoder, embedding projections, and lm_head
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4A16",
    ignore=[
        "lm_head",
        "re:.*vision_tower.*",
        "re:.*audio_tower.*",
        "re:.*embed_vision.*",
        "re:.*embed_audio.*",
    ],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
messages = [
    {"role": "user", "content": "Hello my name is"},
]
text = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
print("==========================================\n\n")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
