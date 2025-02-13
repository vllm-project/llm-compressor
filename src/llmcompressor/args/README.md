# Input arguments for `oneshot`, `train`, `eval` entrypoints

Parsers in `llm-compressor` define the input arguments required for various entry points, including `oneshot`, `train`, and `eval`.

Each entry point (e.g., oneshot) carries out its logic based on the provided input arguments, `model`, `recipe`, and `dataset`.

```python
from llmcompressor import oneshot

model = ...
recipe = ...
dataset = ...
oneshot(model=model, recipe=recipe, dataset=dataset)
```

In addition, users can futher control execution by providing additional arguments. For example, to save the optimized model after completion, the `output_dir` parameter can be specified:

```python
oneshot(
    ..., 
    output_dir=...,
)
```

These input arguments can be overloaded into the function signature and will be parsed using Hugging Face's [argument parser](https://github.com/huggingface/transformers/blob/main/src/transformers/hf_argparser.py). The parsers define the acceptable inputs; therefore any arguments to be passed in must be defined.

`llm-compressor` uses four parsers, located in `llm_compressor/args`:
* ModelArguments
* DatasetArguments
* RecipeArguments
* TrainingArguments


## ModelArguments
Handles model loading and saving. For example, `ModelArguments.model` can be a Hugging Face model identifier or an instance of `AutoModelForCausalLM`. The `save_compressed` flag is a boolean that determines whether the model is saved in compressed safetensors format to minimize disk usage.

## DataArguments
Manages data loading and preprocessing. The dataset argument can specify a Hugging Face dataset stub or a local dataset compatible with [`load_dataset`](https://github.com/huggingface/datasets/blob/3a4e74a9ace62ecd5c9cde7dcb6bcabd65cc7857/src/datasets/load.py#L1905). The preprocessing_func is a callable function that applies custom logic, such as formatting the data using a chat template.

## RecipeArguments
Defines the model recipe. A `recipe` consists of user-defined instructions for optimizing the model. Examples of recipes can be found in the `/examples` directory.

## TrainingArguments
Specifies training parameters based on Hugging Face's [TrainingArguments class](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py). These parameters include settings like learning rate (`learning_rate`), and the optimizer to use (`optim`).

