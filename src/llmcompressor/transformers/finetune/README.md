# Sparse Finetuning

## Launching from Python

```python
from llmcompressor import train

model = "./obcq_deployment"
teacher_model = "Xenova/llama2.c-stories15M"
dataset_name = "open_platypus"
concatenate_data = False
output_dir = "./output_finetune"
recipe = "test_trainer_recipe.yaml"
num_train_epochs=2
overwrite_output_dir = True
splits = {
    "train": "train[:50%]",
}

train(
    model=model,
    distill_teacher=teacher_model,
    dataset=dataset_name,
    output_dir=output_dir,
    recipe=recipe,
    num_train_epochs=num_train_epochs,
    overwrite_output_dir=overwrite_output_dir,
    concatenate_data = concatenate_data,
    splits = splits
)
```

## Additional Configuration

Finetuning arguments are split up into 3 groups:

* ModelArguments: `src/llmcompressor/args/model_arguments.py`
* TrainingArguments: `src/llmcompressor/args/training_arguments.py`
* DatasetArguments: `src/llmcompressor/args/dataset_arguments.py`
* RecipeArguments: `src/llmcompressor/args/recipe_arguments.py`


## Running Multi-Stage Recipes

A recipe can be run stage-by-stage by setting `run_stages` to `True` or calling the 
`llmcompressor.transformers.apply/compress` pathways. Each stage in the recipe should have 
a `run_type` attribute set to either `oneshot` or `train` when running in sequential 
mode.

See [example_alternating_recipe.yaml](../../../../examples/finetuning/example_alternating_recipe.yaml) for an example 
of a staged recipe for Llama. 

test_multi.py
```python
from llmcompressor.transformers import apply
from transformers import AutoModelForCausalLM

model = "../ml-experiments/nlg-text_generation/llama_pretrain-llama_7b-base/dense/training"

dataset_name = "open_platypus"
concatenate_data = False
run_stages=True
output_dir = "./output_finetune_multi"
recipe = "example_alternating_recipe.yaml"
num_train_epochs=1
overwrite_output_dir = True
splits = {
    "train": "train[:95%]",
    "calibration": "train[95%:100%]"
}

apply(
    model_name_or_path=model,
    dataset_name=dataset_name,
    run_stages=run_stages,
    output_dir=output_dir,
    recipe=recipe,
    num_train_epochs=num_train_epochs,
    overwrite_output_dir=overwrite_output_dir,
    concatenate_data = concatenate_data,
    remove_unused_columns = False,
    splits = splits
)

```