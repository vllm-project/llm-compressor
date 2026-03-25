# oneshot

`oneshot` is the primary entrypoint for post-training quantization (PTQ) when your algorithm or scheme requires calibration data. It loads a model through Hugging Face `transformers`, applies recipe-defined modifiers (such as GPTQ, AWQ, SmoothQuant, or QuantizationModifier), and optionally saves the compressed result.

## When to Use

Use `oneshot` when:

- Your quantization algorithm **requires calibration data** (GPTQ, AWQ, SmoothQuant, AutoRound)
- Your scheme uses **static activation quantization** that requires calibration (FP8 per-tensor, INT8 per-tensor, NVFP4 with activations)
- Your model has a **Hugging Face model definition** available via `transformers`

For data-free schemes on models without a transformers definition, or when `oneshot` fails, see [`model_free_ptq`](model-free-ptq.md).

## Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

oneshot(
    model=model,
    recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]),
    output_dir="Meta-Llama-3-8B-Instruct-FP8",
)
```

## Lifecycle

The `oneshot` entrypoint runs three phases:

1. **Preprocessing**
    - Loads the model and tokenizer/processor from the provided identifiers or objects
    - Unties input and output embedding layers if they share tensors
    - Patches `save_pretrained` to support compressed-tensors serialization

2. **Calibration**
    - Wraps the model in a MoE calibration context (if applicable) to ensure all experts receive calibration data
    - Initializes modifiers defined in the recipe via a global `CompressionSession`
    - Runs calibration forward passes through the selected [pipeline](#calibration-pipelines)
    - Finalizes modifiers, applying any post-calibration transformations

3. **Postprocessing**
    - Saves the compressed model, tokenizer/processor, recipe, and config to `output_dir` (if specified)
    - Weights are saved in a compressed SafeTensors format via `compressed-tensors`

## Arguments

### Model Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | `str \| PreTrainedModel` | — | HuggingFace model ID, local path, or a pre-loaded model instance |
| `tokenizer` | `str \| PreTrainedTokenizerBase \| None` | `None` | Tokenizer ID or path. Inferred from `model` if not set |
| `processor` | `str \| ProcessorMixin \| None` | `None` | Processor ID or path (for multimodal models). Inferred from `model` if not set |
| `config_name` | `str \| None` | `None` | Config name or path if different from `model` |
| `precision` | `str` | `"auto"` | Precision to cast model weights to on load (e.g. `"float16"`, `"bfloat16"`, `"auto"`) |
| `trust_remote_code_model` | `bool` | `False` | Allow custom model code from the repository |
| `save_compressed` | `bool` | `True` | Whether to save weights in compressed format |
| `model_revision` | `str` | `"main"` | Model version (branch, tag, or commit) |

### Recipe Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `recipe` | `str \| list \| None` | `None` | Path to a recipe file, a list of paths, or a modifier object / list of modifier objects |
| `recipe_args` | `list[str] \| None` | `None` | Recipe argument overrides in `"key=value"` format |
| `stage` | `str \| None` | `None` | Specific recipe stage to run. Runs all stages if not set |

### Dataset Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `dataset` | `str \| Dataset \| DatasetDict \| DataLoader \| None` | `None` | Dataset name (HuggingFace), a pre-loaded Dataset/DatasetDict, or a PyTorch DataLoader |
| `dataset_config_name` | `str \| None` | `None` | HuggingFace dataset configuration name |
| `dataset_path` | `str \| None` | `None` | Path to a local dataset (JSON, CSV, or DVC) |
| `num_calibration_samples` | `int` | `512` | Number of samples to use for calibration |
| `max_seq_length` | `int` | `384` | Maximum sequence length after tokenization. Longer sequences are truncated |
| `batch_size` | `int` | `1` | Calibration batch size |
| `data_collator` | `str \| Callable` | `"truncation"` | Batch collation strategy. `"truncation"` or `"padding"`, or a custom callable |
| `shuffle_calibration_samples` | `bool` | `True` | Whether to shuffle the dataset before selecting calibration samples |
| `text_column` | `str` | `"text"` | Dataset column to use as text input to the tokenizer/processor |
| `concatenate_data` | `bool` | `False` | Whether to concatenate samples to fill `max_seq_length` |
| `streaming` | `bool` | `False` | Stream data from a cloud-hosted dataset |
| `preprocessing_num_workers` | `int \| None` | `None` | Number of workers for dataset preprocessing |
| `dataloader_num_workers` | `int` | `0` | Number of workers for the DataLoader. Set to 2+ for faster loading if RAM allows |
| `moe_calibrate_all_experts` | `bool` | `True` | Route all tokens through all experts during calibration. Required for accurate MoE quantization |
| `min_tokens_per_module` | `float \| None` | `None` | Minimum fraction of tokens a module must receive. Logs a warning if unmet. Mainly relevant for MoE models |

### Pipeline Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `pipeline` | `str \| None` | `"independent"` | Calibration pipeline to use. See [Calibration Pipelines](#calibration-pipelines) |
| `sequential_targets` | `list[str] \| None` | `None` | Layer targets for the sequential pipeline (typically a single decoder layer class). Defaults to `no_split_modules` from the HF model definition |
| `sequential_offload_device` | `str` | `"cpu"` | Device to offload intermediate activations between sequential layers. Use `"cuda:1"` if a second GPU is available |
| `quantization_aware_calibration` | `bool` | `True` | Apply quantization during the calibration forward pass in the sequential pipeline |
| `sequential_prefetch` | `bool` | `False` | Prefetch the next batch in a background thread during sequential pipeline calibration |

### Miscellaneous Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `output_dir` | `str \| None` | `None` | Directory to save the compressed model. Nothing is saved if `None` |
| `log_dir` | `str \| None` | `None` | Directory to write timestamped log files. Nothing is logged to file if `None` |

## Calibration Pipelines

The `pipeline` argument controls how calibration forward passes are run through the model.

| Pipeline | Description | Best For |
|----------|-------------|----------|
| `independent` | Each modifier manages its own forward passes independently *(default)* | Most use cases |
| `sequential` | Runs calibration layer-by-layer, offloading intermediate activations between layers | Large models that don't fit in GPU memory |
| `datafree` | Runs initialization and finalization without any forward passes | Data-free weight-only quantization |
| `basic` | Single set of forward passes shared across all modifiers | Simple post-hoc calibration |

## Examples

### FP8 Data-Free Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

oneshot(
    model=model,
    recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]),
    output_dir="Meta-Llama-3-8B-Instruct-FP8",
)
```

### GPTQ W4A16

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

oneshot(
    model=model,
    dataset="HuggingFaceH4/ultrachat_200k",
    recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
    num_calibration_samples=512,
    max_seq_length=2048,
    output_dir="Meta-Llama-3-8B-Instruct-W4A16-GPTQ",
)
```

### MoE Model with All-Expert Calibration

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modeling.llama4 import SequentialLlama4TextMoe  # noqa: F401
from llmcompressor.modifiers.quantization import QuantizationModifier

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")

oneshot(
    model=model,
    dataset="HuggingFaceH4/ultrachat_200k",
    recipe=QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            "re:.*lm_head",
            "re:.*self_attn",
            "re:.*router",
            "re:.*vision_model.*",
            "re:.*multi_modal_projector.*",
            "Llama4TextAttention",
        ],
    ),
    num_calibration_samples=20,
    max_seq_length=2048,
    moe_calibrate_all_experts=True,
    output_dir="Llama-4-Scout-17B-NVFP4",
)
```

## Saving

The recommended way to save is via the `output_dir` argument, which automatically saves the model weights in compressed SafeTensors format along with the tokenizer/processor, recipe, and config:

```python
oneshot(..., output_dir="./my-compressed-model")
```

Alternatively, you can save manually after the call:

```python
model = oneshot(model=model, recipe=recipe)
model.save_pretrained("./my-compressed-model", save_compressed=True)
tokenizer.save_pretrained("./my-compressed-model")
```

For more details on save options, see [Saving a Compressed Model](../saving_a_model.md).
