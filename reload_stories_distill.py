import os, shutil
from llmcompressor.core import create_session
from llmcompressor.transformers import (
    SparseAutoModelForCausalLM,
    oneshot, train
)

output_dir = "./distill_out"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
dataset = "open_platypus"
concatenate_data = False
splits = "train[:50%]"
max_steps = 2
num_calibration_samples = 64
recipe_str = "tests/llmcompressor/transformers/finetune/test_finetune_recipe.yaml"

# base
model = SparseAutoModelForCausalLM.from_pretrained(
    "Xenova/llama2.c-stories15M", device_map="auto"
    #"./oneshot_out", device_map="auto"
)
distill_teacher = SparseAutoModelForCausalLM.from_pretrained(
    #"Xenova/llama2.c-stories15M", device_map="auto"
    "./oneshot_out", device_map="auto"
)

# distill
with create_session():
    train(
        model=model,
        distill_teacher=distill_teacher,
        dataset=dataset,
        output_dir=output_dir,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe_str,
        concatenate_data=concatenate_data,
        splits=splits,
        max_steps=max_steps,
    )

# load
model = SparseAutoModelForCausalLM.from_pretrained(
    output_dir, device_map="auto"
)