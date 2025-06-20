# `int8` Weight and Activation Quantization

`llm-compressor` supports quantizing weights and activations to `int` for memory savings and inference acceleration with `vLLM`

> `int8` compuation is supported on Nvidia GPUs with compute capability > 7.5 (Turing, Ampere, Ada Lovelace, Hopper).

## Installation

To get started, install:

```bash
pip install llmcompressor
```

## Quickstart

The example includes an end-to-end script for applying the quantization algorithm.

```bash
python3 llama3_example.py
```

The resulting model `Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token` is ready to be loaded into vLLM.

## Code Walkthough

Now, we will step though the code in the example. There are four steps:
1) Load model
2) Prepare calibration data
3) Apply quantization
4) Evaluate accuracy in vLLM

### 1) Load Model

Load the model using `AutoModelForCausalLM` for handling quantized saving and loading.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2) Prepare Calibration Data

Prepare the calibration data. When quantizing activations of a model to `int8`, we need some sample data to estimate the activation scales. As a result, it is very useful to use calibration data that closely matches the type of data used in deployment. If you have fine-tuned a model, using a sample of your training data is a good idea.

In our case, we are quantizing an Instruction tuned generic model, so we will use the `ultrachat` dataset. Some best practices include:
* 512 samples is a good place to start (increase if accuracy drops)
* 2048 sequence length is a good place to start
* Use the chat template or instrucion template that the model is trained with

```python
from datasets import load_dataset

NUM_CALIBRATION_SAMPLES=512
MAX_SEQUENCE_LENGTH=2048

# Load dataset.
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

# Preprocess the data into the format the model is trained with.
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False,)}
ds = ds.map(preprocess)

# Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)
```

### 3) Apply Quantization

With the dataset ready, we will now apply quantization.

We first select the quantization algorithm. For W8A8, we want to:
* Run SmoothQuant to make the activations easier to quantize
* Quantize the weights to 8 bits with channelwise scales using GPTQ
* Quantize the activations with dynamic per token strategy

> See the `Recipes` documentation for more information on recipes

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# Configure the quantization algorithms to run.
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W8A8-Dynamic-Per-Token"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

We have successfully created an `w8a8` model with weights and activations quantized to 8-bit integers!

### 4) Evaluate Accuracy

With the model created, we can now load and run in vLLM (after installing).

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token")
```

We can evaluate accuracy with `lm_eval` (`pip install lm_eval==v0.4.3`):
> Note: quantized models can be sensitive to the presence of the `bos` token. `lm_eval` does not add a `bos` token by default, so make sure to include the `add_bos_token=True` argument when running your evaluations.

Run the following to test accuracy on GSM-8K:

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

We can see the resulting scores look good!

```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.752|±  |0.0274|
|     |       |strict-match    |     5|exact_match|↑  |0.756|±  |0.0272|
```

### Questions or Feature Request?

Please open up an issue on `vllm-project/llm-compressor`
