# Compression and Fine-tuning Entrypoint

## Oneshot

An ideal compression technique reduces memory footprint while maintaining accuracy. One-shot in LLM-Compressor supports faster inference on vLLM by applying post-training quantization (PTQ) or sparsification.

### PTQ
PTQ is performed to reduce the precision of quantizable weights (e.g., linear layers) to a lower bit-width. Supported formats are:
- [W4A16](../../../examples/quantization_w4a16/README.md)
- [W8A8-INT8](../../../examples/quantization_w8a8_int8/README.md)
- [W8A8-FP8](../../../examples/quantization_w8a8_fp8/README.md)

### Sparsification
Sparsification reduces model complexity by pruning selected weight values to zero while retaining essential weights in a subset of parameters. Supported formats include:
-  [2:4-Sparsity with FP4 Weight](../../../examples/quantization_2of4_sparse_w4a16/README.md)
-  [2:4-Sparsity with FP8 Weight, FP8 Input Activation](../../../examples/sparse_2of4_quantization_fp8/README.md)

## Code

Example scripts for all the above formats are located in the [examples](../../../examples/) folder. The [W8A8-FP8](../../../examples/quantization_w8a8_fp8/llama3_example.py) example is shown below: 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Define the model to compress
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load the model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Define the recipe, scheme="FP8_DYNAMIC" compresses to W8A8, which is
# FP8 channel-wise for weight, and FP8 dynamic per token activation
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

# compress the model
oneshot(model=model, recipe=recipe)
```

### Input Arguments
`oneshot` only accepts arguments defined in `src/llmcompressor/args`, which are dataclasses categorized into [`ModelArguments`](../../llmcompressor/args/model_arguments.py), [`DatasetArguments`](../../llmcompressor/args/dataset_arguments.py) and [`RecipeArguments`](../../llmcompressor/args/recipe_arguments.py). If an undefined input argument is provided, an error will be raised.

The high-level description of the argument parser is as follows:

- `ModelArguments`: Arguments for loading and configuring a pretrained model
    (e.g., `AutoModelForCausalLM`).
- `DatasetArguments`: Arguments for dataset-related configurations, such as
    calibration dataloaders.
- `RecipeArguments`: Arguments for defining and configuring recipes that specify
    parameters for compression.

For more information, please check the [README.md](../../llmcompressor/args/README.md) in `src/llmcompressor/args`.

### Saving the Compressed Model

To save the compressed model, the recommended approach is to specify `output_dir` as the desired destination directory. By default, the model will be saved in a compressed format, reducing its disk space usage upon saving.

```python
oneshot(
    ...,
    output_dir="./oneshot_model", # Automatically save the safetensor, config, recipe. Weights are saved in a compressed format
)
```    


### Lifecycle

The oneshot calibration lifecycle consists of three steps:
1. **Preprocessing**:
    - Instantiates a pretrained model and tokenizer/processor.
    - Ensures input and output embedding layers are untied if they share
        tensors.
    - Patches the model to include additional functionality for saving with
        quantization configurations.
2. **Oneshot Calibration**:
    - Compresses the model based on the recipe (instructions for optimizing the model). The 
        recipe defines the `Modifiers` (e.g., `GPTQModifier`, `SparseGPTModifier`) to apply, which
        contain logic how to quantize or sparsify a model. 
3. **Postprocessing**:
    - Saves the model, tokenizer/processor, and configuration to the specified
        `output_dir`.

This will automatically save the model weights to a compressed SafeTensors format. The tokenizer/processor, recipe, and the configuration file will also be saved.

## Train / Finetune
Compressed models can be trained to improve accuracy. Training is carried out using HuggingFace's Trainer.

### Finetuning a Compressed Model
LLM-Compressor supports fine-tuning of quantized, sparsified, and sparse-quantized models. It offers both standard fine-tuning, knowledge distillation and SFT Trainer.

## Code

### Finetuning

A compressed model generated using `oneshot` is saved to disk in a compressed format. To load it, the model must be decompressed using `CompressedTensorsConfig` with `AutoModelForCausalLM`. If the above `oneshot` example script was executed and the compressed model was saved to `./oneshot_model`, the following code is used to perform fine-tuning:


```python
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor import create_session, train

# The saving directory
output_dir = "./oneshot_model"

# The model to train
model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    quantization_config=CompressedTensorsConfig(run_compressed=False),
)

dataset = "open_platypus"  # Define dataset to use for kd
output_dir = "./finetuned_model"
splits = "train[:50%]"  # Use 50% of the training data
max_steps = (
    25  # Number of training steps (updates) before stopping the training process
)
num_calibration_samples = 8  # Number of workers processing datasets in parallel

# Create an isolated session independent from the previous runs
with create_session():
    train(
        model=model,  # The model to finetune
        dataset=dataset,  # The data to carry out finetuning
        output_dir=output_dir,  # The output directory to save
        num_calibration_samples=num_calibration_samples,  # The number of workers to carry out dataset processing
        splits=splits,  # The dataset key and percentage of samples to use
        max_steps=max_steps,  # The total number of iterations to carry out training
    )
```


### Knowledge Distillation

To perform knowledge distillation, a teacher model and a student model (the compressed model) must be defined. The loss between the student and the teacher can be specified in the recipe by defining the `comparison` key. In this case, KL divergence is used to compare the output distributions of the student and the teacher.
Comparisons are defined in `/src/llmcompressor/modifiers/distillation/utils/pytorch/kd_factory.py`.

```python
# Define the teacher model
distill_teacher = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",  
)

# Define the recipe, use knowledge distillation modifier and target the `model.layers` using a regex with
recipe = r"""
kd_stage:
  distillation_modifiers:
    OutputDistillationModifier:
        targets: ["re:model.layers.\\d+$"]
        comparison: "kl_divergence"
        start: 0
        orig_scale: 1.0
        distill_scale: 1.0
"""

# Create an isolated session from the previous runs
with create_session():
    train(
        ...
        distill_teacher=distill_teacher,    # The teacher model
        recipe=recipe,                      # The recipe to use
    )

```

The output terminal will provide the sparsification, quantization and training metrics:

```bash
2025-02-25T18:39:08.984855-0500 | log_model_sparsification | INFO - There are 8033013760 prunable params which have 0.02% avg sparsity.
2025-02-25T18:39:08.987302-0500 | log_model_sparsification | INFO - There are 8033013760 quantizable params, with a quantization percentage of 86.88%.
***** train metrics *****
  epoch                    =      0.016
  perplexity               =     1.5422
  total_flos               =  3221945GF
  train_loss               =     0.4332
  train_runtime            = 0:03:53.39
  train_samples            =      12463
  train_samples_per_second =      0.857
  train_steps_per_second   =      0.107
```

### End-to-end Script 
The end-to-end script for carrying out `oneshot` for `W8A8-FP8` and then knowledge distillation is shown below:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# The directory for saving
oneshot_output_dir = "./oneshot_model"

# Load the model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Define the recipe. `scheme="FP8_DYNAMIC"` compresses to W8A8-FP8, which is
# FP8 channel-wise for weight, and FP8 dynamic per token activation
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

# compress the model
oneshot(model=model, recipe=recipe, output_dir=oneshot_output_dir)

from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor import create_session, train

# Student model
model = AutoModelForCausalLM.from_pretrained(
    oneshot_output_dir,
    quantization_config=CompressedTensorsConfig(run_compressed=False),
)

dataset = "open_platypus"  # Define dataset to use for knowledge distillation
finetune_output_dir = "./finetuned_model"  # The output saving directory
splits = "train[:50%]"  # Use 50% of the training data
max_steps = (
    25  # The number of training steps (updates) before stopping the training process
)
num_calibration_samples = 8  # The number of workers processing datasets in parallel

# Define teacher model
distill_teacher = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
)

# Define the recipe, use knowledge distillation modifier and target the `model.layers` using a regex with
# KL divergence comparison
recipe = r"""
kd_stage:
  distillation_modifiers:
    OutputDistillationModifier:
        targets: ["re:model.layers.\\d+$"]
        comparison: "kl_divergence"
        start: 0
        orig_scale: 1.0
        distill_scale: 1.0
"""

# Create an isolated session from the previous runs
with create_session():
    train(
        model=model,  # The student model
        dataset=dataset,  # The data to carry out finetuning
        output_dir=finetune_output_dir,  # Output directory to save
        num_calibration_samples=num_calibration_samples,  # The number of workers to carry out dataset processing
        splits=splits,  # The percentage of the subsets of a dataset to use
        max_steps=max_steps,  # The number of training steps
        distill_teacher=distill_teacher,  # The teacher model
        recipe=recipe,  # The recipe to use
    )
```

### SFT Trainer

TRL's SFT Trainer can be used for sparse fine-tuning or applying sparse knowledge distillation. Examples are available in the `examples/` folder.

- [Sparse-fine-tune a 50% sparse Llama-7b model](../../../examples/trl_mixin/README.md)
- [Sparse-fine-tune a 50% sparse Llama-7b model using knowledge distillation](../../../examples/trl_mixin/README.md)