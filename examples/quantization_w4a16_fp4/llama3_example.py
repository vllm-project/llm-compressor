from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

from llmcompressor.transformers.compression.helpers import calculate_offload_device_map

#MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

# Load model.

from compressed_tensors.utils import offloaded_dispatch

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map=None, torch_dtype="auto"
)
offloaded_dispatch(model, execution_device=torch.device("cuda"))  # model is now offloaded
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

for layer in model.model.layers:
    print(layer.self_attn.q_proj.weight.device)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 16 via ptq
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4A16", ignore=["lm_head"])

# Apply quantization.
oneshot(model=model, recipe=recipe)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=25)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.split("/")[1] + "-NVFP4A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
