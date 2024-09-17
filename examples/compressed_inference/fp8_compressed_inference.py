from transformers import AutoTokenizer

from llmcompressor.transformers import SparseAutoModelForCausalLM

"""
This example covers how to load a quantized model in compressed mode. By default,
SparseAutoModelForCausalLM will decompress the whole model on load resulting in no
memory savings from quantization. By setting the `run_compressed` kwarg to True, the
model will remain compressed in memory on load, saving memory during inference at the
cost of increased runtime

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

# set run_compressed=True to enable running in compressed mode
compressed_model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_STUB, torch_dtype="auto", device_map="cuda:0", run_compressed=True
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
