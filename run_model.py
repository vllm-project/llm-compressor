from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "nm-testing/TinyLlama-1.1B-Chat-v1.0-w4a16-sym-awq-e2e"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
#dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")
