from transformers import AutoModelForCausalLM, AutoTokenizer

"""
This example covers how to load a quantized model using AutoModelForCausalLM.

During inference, each layer will be decompressed as needed before the forward pass.
This saves memory as only a single layer is ever uncompressed at a time, but increases
runtime as we need to decompress each layer before running the forward pass

"""

# any model with the "compressed-tensors" quant_method and "compressed"
# quantization_status in the quantization config is supported
MODEL_STUB = "nm-testing/tinyllama-fp8-dynamic-compressed"

SAMPLE_INPUT = [
    "I love quantization because",
    "What is the capital of France?",
    "def fibonacci(n):",
]

compressed_model = AutoModelForCausalLM.from_pretrained(
    MODEL_STUB,
    torch_dtype="auto",
    device_map="cuda:0",
)

# tokenize the sample data
tokenizer = AutoTokenizer.from_pretrained(MODEL_STUB)
inputs = tokenizer(SAMPLE_INPUT, return_tensors="pt", padding=True).to(
    compressed_model.device
)

# run the compressed model and decode the output
output = compressed_model.generate(**inputs, max_length=50)
print("========== SAMPLE GENERATION ==============")
text_output = tokenizer.batch_decode(output)
for sample in text_output:
    print(sample)
