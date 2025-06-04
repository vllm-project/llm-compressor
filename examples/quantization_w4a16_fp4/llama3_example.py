from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_ID = "Qwen/Qwen3-30B-A3B"
# Load model.
device_map = calculate_offload_device_map(MODEL_ID, reserve_for_hessians=False, num_gpus=2)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map=device_map, torch_dtype="auto"
)

"""
count = 0
for l in model.model.layers:
    output = l.self_attn.q_proj.weight.device
    output = str(output)
    if "cuda" in output:
        count += 1
print(count)
"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 16 via ptq
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4A16", ignore=["lm_head"])

# Apply quantization.
oneshot(model=model, recipe=recipe)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")


# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.split("/")[1] + "-NVFP4A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
