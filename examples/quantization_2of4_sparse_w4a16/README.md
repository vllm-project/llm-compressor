# `int4` Weight Quantization of a 2:4 Sparse Model

`llm-compressor` supports quantizing weights while maintaining sparsity patterns for memory savings and inference acceleration with `vLLM`

> `2:4 sparisty + int4/int8` mixed precision computation is supported in vLLM on Nvidia capability > 8.0 (Ampere, Ada Lovelace, Hopper).

## NOTE: 
Fine tuning can require more steps than is shown in the example.
See the Axolotl integration blog post for best fine tuning practices
https://developers.redhat.com/articles/2025/06/17/axolotl-meets-llm-compressor-fast-sparse-open


## Installation

To get started, install:

```bash
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

## Quickstart

The example includes an end-to-end script for applying the quantization algorithm.

```bash
python3 llama7b_sparse_w4a16.py
```


# Creating a Sparse Quantized Llama7b Model

This example uses LLMCompressor and Compressed-Tensors to create a 2:4 sparse and quantized Llama2-7b model.
The model is calibrated and trained with the ultachat200k dataset.
At least 75GB of GPU memory is required to run this example.

Follow the steps below, or to run the example as `python examples/quantization_2of4_sparse_w4a16/llama7b_sparse_w4a16.py`

## Step 1: Select a model, dataset, and recipe
In this step, we select which model to use as a baseline for sparsification, a dataset to
use for calibration and finetuning, and a recipe.

Models can reference a local directory, or a model in the huggingface hub.

Datasets can be from a local compatible directory or the huggingface hub.

Recipes are YAML files that describe how a model should be optimized during or after training.
The recipe used for this flow is located in [2of4_w4a16_recipe.yaml](./2of4_w4a16_recipe.yaml).
It contains instructions to prune the model to 2:4 sparsity, run one epoch of recovery finetuning,
and quantize to 4 bits in one show using GPTQ.

```python
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot, train

# load the model in as bfloat16 to save on memory and compute
model_stub = "neuralmagic/Llama-2-7b-ultrachat200k"
model = AutoModelForCausalLM.from_pretrained(model_stub, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_stub)

# uses LLM Compressor's built-in preprocessing for ultra chat
dataset = "ultrachat-200k"

# Select the recipe for 2 of 4 sparsity and 4-bit activation quantization
recipe = "2of4_w4a16_recipe.yaml"

# save location of quantized model
output_dir = "output_llama7b_2of4_w4a16_channel"
output_path = Path(output_dir)

# set dataset config parameters
splits = {"calibration": "train_gen[:5%]", "train": "train_gen"}
max_seq_length = 512
num_calibration_samples = 512

# set training parameters for finetuning
# increase num_train_epochs for longer training
num_train_epochs = 0.01
logging_steps = 500
save_steps = 5000
gradient_checkpointing = True  # saves memory during training
learning_rate = 0.0001
bf16 = False  # using full precision for training
lr_scheduler_type = "cosine"
warmup_ratio = 0.1
preprocessing_num_workers = 8
```

## Step 2: Run `sparsification`, `fine-tuning`, and `quantization`
The compression process now runs in three stages: sparsification, fine-tuning, and quantization.
Each stage saves the intermediate model outputs to the `output_llama7b_2of4_w4a16_channel` directory.

```python
from llmcompressor import oneshot, train
from pathlib import Path

output_dir = "output_llama7b_2of4_w4a16_channel"
output_path = Path(output_dir)

# 1. Oneshot sparsification: apply pruning
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    splits=splits,
    num_calibration_samples=num_calibration_samples,
    preprocessing_num_workers=preprocessing_num_workers,
    output_dir=output_dir,
    stage="sparsity_stage",
)

# 2. Sparse fine-tuning: improve accuracy on pruned model
train(
    model=output_path / "sparsity_stage",
    dataset=dataset,
    recipe=recipe,
    splits=splits,
    num_calibration_samples=num_calibration_samples,
    preprocessing_num_workers=preprocessing_num_workers,
    bf16=bf16,
    max_seq_length=max_seq_length,
    num_train_epochs=num_train_epochs,
    logging_steps=logging_steps,
    save_steps=save_steps,
    gradient_checkpointing=gradient_checkpointing,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=warmup_ratio,
    output_dir=output_dir,
    stage="finetuning_stage",
)

# 3. Oneshot quantization: compress model weights to lower precision
quantized_model = oneshot(
    model=output_path / "finetuning_stage",
    dataset=dataset,
    recipe=recipe,
    splits=splits,
    num_calibration_samples=num_calibration_samples,
    preprocessing_num_workers=preprocessing_num_workers,
    output_dir=output_dir,
    stage="quantization_stage",
)
# skip_sparsity_compression_stats is set to False
# to account for sparsity in the model when compressing
quantized_model.save_pretrained(
    f"{output_dir}/quantization_stage", skip_sparsity_compression_stats=False
)
tokenizer.save_pretrained(f"{output_dir}/quantization_stage")

```

### Custom Quantization
The current repo supports multiple quantization techniques configured using a recipe. Supported strategies are tensor, group, and channel.

The recipe (`2of4_w4a16_recipe.yaml`) uses channel-wise quantization (`strategy: "channel"`).
To change the quantization strategy, edit the recipe file accordingly:

Use `tensor` for per-tensor quantization
Use `group` for group-wise quantization and specify the group_size parameter (e.g., 128)
See `2of4_w4a16_group-128_recipe.yaml` for a group-size example
