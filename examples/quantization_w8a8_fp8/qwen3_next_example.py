from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"

# Load model.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

recipe = QuantizationModifier(
    targets=["Linear"],
    scheme="FP8_DYNAMIC",
    ignore=[
        "lm_head",
        "re:.*mlp.gate$",
        "re:.*mlp.shared_expert_gate$",
        "re:.*linear_attn.*",
    ],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
