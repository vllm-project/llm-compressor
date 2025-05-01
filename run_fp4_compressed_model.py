from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig
import torch

config = CompressedTensorsConfig(run_compressed=False)

MODEL_ID = "nm-testing/llama2.c-stories110M-FP4"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto", quantization_config=config)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))

