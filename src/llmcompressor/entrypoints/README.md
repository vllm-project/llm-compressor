# Compression and Fine-tuning Entrypoint

## Oneshot

An ideal compression technique reduces memory footprint while maintaining accuracy. One-shot in LLM-Compressor supports faster inference on vLLM by applying post-training quantization (PTQ) or sparsification.

### PTQ
PTQ is performed to reduce the precision of quantizable weights (e.g., linear layers) to a lower bit-width. Supported formats are:
- W4A16
- W8A8-INT8 
- W8A8-FP8

### Sparsification
Sparsification reduces model complexity by pruning selected weight values to zero while retaining essential weights in a subset of parameters. Supported formats include:
-  2:4-Sparsity with FP8 Weight, FP8 Input Activation


## Code

Example scripts for all the above formats are located in the [examples](../../../examples/) folder. A [W8A8-FP8](../../../examples/quantization_w8a8_fp8/llama3_example.py) example is shown below: 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

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
    compression methods.

For more information, please check the [README.md](../../llmcompressor/args/README.md) in `src/llmcompressor/args`.


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

### Saving a Compressed Model

To save an compressed model, the recommended approach is to specify `output_dir` in the input argument. For example, to save the model in the `./oneshot_model` directory,

```python3
oneshot(
    ...,
    output_dir="./oneshot_model",
)
```    

This will automatically save the model in the SafeTensors format, along with the tokenizer/processor, recipe, and the configuration file.


## Train
Compressed models can be trained to imporve accuracy. LLM Compressor uses HuggingFace's [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer) for training.

### Finetuning an Compressed Model
LLM-Compressor supports fine-tuning of quantized, sparsified, or sparse-quantized models. It offers both vanilla fine-tuning and knowledge distillation.

## Code
A compressed model generated from `oneshot` is saved to disk in a compressed format. To load it, the model must be decompressed using `CompressedTensorsConfig` with `AutoModelForCausalLM`. Suppose the above `oneshot` example script was executed and the compressed model was saved to `./oneshot_model`. Then use the following code to carry out fine-tuning:

```python
from transformers.utils.quantization_config import CompressedTensorsConfig

output_dir="./oneshot_model"
model = AutoModelForCausalLM.from_pretrained(
    output_dir,
    device_map="auto",
    quantization_config=self.quantization_config,
)

dataset = "open_platypus"
concatenate_data = False
output_dir "./finetuned_model"
splits = "train[:50%]"
max_steps = 25
num_calibration_samples = 64

with create_session():
    train(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        num_calibration_samples=num_calibration_samples,
        concatenate_data=concatenate_data,
        splits=splits,
        max_steps=max_steps,
    )

```

To carry out knowledge distillation, a teacher model and a student model (the compressed model) must be defined. Use the above training with 

```python

distill_teacher = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", 
    device_map="auto",
)

recipe = """
kd_stage:
  distillation_modifiers:
    OutputDistillationModifier:
        targets: ["re:model.layers.\\d+$"]
        comparison: "square_head"
        start: 0
        orig_scale: 1.0
        distill_scale: 1.0
"""

with create_session():
    train(
        model=model,
        distill_teacher=distill_teacher,
        recipe=recipe,
        ...
    )
```





