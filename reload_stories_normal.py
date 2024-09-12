import os, shutil
from llmcompressor.transformers import SparseAutoModelForCausalLM

output_dir = "./my_model"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# base
model = SparseAutoModelForCausalLM.from_pretrained("Xenova/llama2.c-stories15M", device_map="auto", torch_dtype="auto")

# save
model.save_pretrained(
    output_dir,
    save_compressed=True,
    safe_serialization=True,  # False:=pytorch_model.bin, True:=model.safetensors
)

# load normal
model = SparseAutoModelForCausalLM.from_pretrained(
    output_dir, device_map="auto"
)
print(model)