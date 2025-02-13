# LLM Compressor Entrypoints

## Oneshot

Model optimizations compress models while preserving accuracy. One-shot in LLM-Compressor supports faster inference on vLLM by applying post-training quantization (PTQ) or sparsification

### PTQ
PTQ is performed to reduce the precision of quantizable weights (e.g., linear layers) to a lower bit-width. Supported formats are:

#### W4A16
- Uses GPTQ to compress weights to 4 bits. 
- Useful speed ups in low QPS regimes with more weight compression. 
- Recommended for any GPUs types. 
#### W8A8-INT8 
- Uses channel-wise quantization to compress weights to 8 bits, and uses dynamic per-token quantization to compress activations to 8 bits. 
- Useful for speed ups in high QPS regimes or offline serving on vLLM. 
- Recommended for NVIDIA GPUs with compute capability <8.9 (Ampere, Turing, Volta, Pascal, or older). 
#### W8A8-FP8
- Uses channel-wise quantization to compress weights to 8 bits, and uses dynamic per-token quantization to compress activations to 8 bits. 
- Useful for speed ups in high QPS regimes or offline serving on vLLM. 
- Recommended for NVIDIA GPUs with compute capability >8.9 (Hopper and Ada Lovelace). 

### Sparsification
Sparsification reduces model complexity by pruning selected weight values to zero while retaining essential weights in a subset of parameters. Supported formats include:

#### 2:4-Sparsity
- Uses semi-structured sparsity, where for every four contiguous weights in a tensor, two are set to zero. 
- Useful for efficiently computing sparse matrix multiplications using NVIDIA Sparse Tensor Cores. To preserve as much accuracy from the base model, usually a sparse-finetune step is added. 
- Recommended for NVIDIA architecture from Amphere onwards, supported on GPUs with Sparse Tensor Cores

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
    optimization actions.

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
    - Optimizes the model based on the recipe (instructions for optimizing the model). The 
        recipe defines the `Modifiers` (e.g., `GPTQModifier`, `SparseGPTModifier`) to apply, which
        contain logic how to quantize or sparsify a model. 
3. **Postprocessing**:
    - Saves the model, tokenizer/processor, and configuration to the specified
        `output_dir`.

### Saving an Optimized Model

To save an optimized model, the recommended approach is to specify `output_dir` in the input argument. For example, to save the model in the `./oneshot_model` directory,

```python3
oneshot(
    ...,
    output_dir="./oneshot_model",
)
```    

This will automatically save the model in the SafeTensors format, along with the tokenizer/processor, recipe, and the configuration file.
