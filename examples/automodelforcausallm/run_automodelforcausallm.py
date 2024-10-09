from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "nm-testing/tinyllama-w8a8-compressed-hf-quantizer"

# Use the AutoModelForCausalLM to run the model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
