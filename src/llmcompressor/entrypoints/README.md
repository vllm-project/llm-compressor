# Compression Entrypoints

## Oneshot
An ideal compression technique reduces memory footprint while maintaining accuracy. One-shot in LLM-Compressor supports faster inference on vLLM by applying post-training quantization (PTQ) or sparsification.

### PTQ
PTQ is performed to reduce the precision of quantizable weights (e.g., linear layers) to a lower bit-width. 
A complete list of formats can be found here: https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/compression_schemes/

### Sparsification
Sparsification reduces model complexity by pruning selected weight values to zero while retaining essential weights in a subset of parameters. Supported formats include:
-  [2:4-Sparsity with FP8 Weight Activation Quantization](../../../examples/sparse_2of4_quantization_fp8/README.md)

### Example

Example scripts for all the above formats are located in the [examples](../../../examples/) folder. The [W8A8-FP8](../../../examples/quantization_w8a8_fp8/llama3_example.py) example is shown below: 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Define the model to compress
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load the model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
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

## Model-Free PTQ
For certain cases, it may be beneficial to consider the `model_free_ptq` entrypoint such as when a model definition is lacking or if the `oneshot` entrypoint fails. 
`model_free_ptq` can be applied for schemes that do not require data, such as Round-To-Nearest with FP8 or NVFP4A16. Examples applying the entrypoint can be found
here: https://github.com/vllm-project/llm-compressor/tree/main/examples/model_free_ptq.

### Finetuning

As of LLM Compressor v0.9.0, training support has been deprecated. To apply finetuning to your model, such as in the case of sparse-finetuning, Axolotl training can be applied. A step-by-step guide explaining how to apply the Axolotl integration can be found here: https://developers.redhat.com/articles/2025/06/17/axolotl-meets-llm-compressor-fast-sparse-open# as well as in the Axolotl documentation: https://docs.axolotl.ai/docs/custom_integrations.html#llmcompressor.