import os, shutil
from llmcompressor.core import create_session
from llmcompressor.transformers import (
    SparseAutoModelForCausalLM,
    oneshot,
)

output_dir = "./oneshot_out"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
recipe_str = "tests/llmcompressor/transformers/obcq/recipes/test_tiny2.yaml"
dataset = "open_platypus"
concatenate_data = False
num_calibration_samples = 64
splits = {"calibration": "train[:10%]"}


# base
model = SparseAutoModelForCausalLM.from_pretrained(
    "Xenova/llama2.c-stories15M", device_map="auto"
)

# save oneshot
with create_session():
    oneshot(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
    )

# load oneshot
model = SparseAutoModelForCausalLM.from_pretrained(
    output_dir, device_map="auto"
)
print(model)
