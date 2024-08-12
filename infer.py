from llmcompressor.transformers import SparseAutoModelForCausalLM
from transformers import AutoTokenizer

model_name = "actorder20240812_220108"
model = SparseAutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda") 
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0]))
