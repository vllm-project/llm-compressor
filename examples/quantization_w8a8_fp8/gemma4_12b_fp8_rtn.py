# Quantize gemma-4-12B-it to FP8 using RTN (round-to-nearest)
#
# Evaluate with e.g.:
#   lm_eval --model vllm \
#     --model_args "pretrained=gemma-4-12B-it-FP8-RTN,\
#       dtype=auto,max_model_len=4096,add_bos_token=True,\
#       gpu_memory_utilization=0.85" \
#     --tasks gsm8k_platinum --num_fewshot 5 \
#     --apply_chat_template --batch_size auto

from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "google/gemma-4-12B-it"
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "lm_head",
        "re:.*embed_vision.*",
        "re:.*embed_audio.*",
        "re:.*vision_embedder.*",
    ],
)

oneshot(
    model=model,
    recipe=recipe,
)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
messages = [{"role": "user", "content": "The reason the sky is blue is because"}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
input_ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(processor.tokenizer.decode(output[0]))
print("==========================================\n\n")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
