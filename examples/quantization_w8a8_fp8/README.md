# `fp8` Weight and Activation Quantization

`llm-compressor` supports quantizing weights and activations to `fp8` for memory savings and inference acceleration with `vLLM`

> `fp8` compuation is supported on Nvidia GPUs with compute capability > 8.9 (Ada Lovelace, Hopper).

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
python3 llama3_example.py
```

The resulting model `Meta-Llama-3-8B-Instruct-W8A8-FP8` is ready to be loaded into vLLM.

## Code Walkthough

Now, we will step though the code in the example. There are four steps:
1) Load model
2) Prepare calibration data
3) Apply quantization
4) Evaluate accuracy in vLLM

### 1) Load Model

Load the model using `SparseAutoModelForCausalLM`, which is a wrapper around `AutoModel` for handling quantized saving and loading. Note that `SparseAutoModel` is compatible with `accelerate` so you can load your model onto multiple GPUs if needed.

```python
from llmcompressor.transformers import SparseAutoModelForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2) Prepare Calibration Data

Prepare the calibration data. When quantizing activations of a model to `fp8`, we need some sample data to estimate the activation scales. As a result, it is very useful to use calibration data that closely matches the type of data used in deployment. If you have fine-tuned a model, using a sample of your training data is a good idea.

In our case, we are quantizing an Instruction tuned generic model, so we will use the `ultrachat` dataset. Some best practices include:
* 512 samples is a good place to start (increase if accuracy drops)
* 2048 sequence length is a good place to start
* Use the chat template or instrucion template that the model is trained with

```python
from datasets import load_dataset

NUM_CALIBRATION_SAMPLES=512
MAX_SEQUENCE_LENGTH=2048

# Load dataset.
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

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

We first select the quantization algorithm. In our case, we will apply the default recipe for `fp8` (which uses static-per-tensor weights and static-per-tensor activations) to all linear layers.
> See the `Recipes` documentation for more information on making complex recipes

```python
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure the quantization algorithm to run.
recipe = QuantizationModifier(targets="Linear", scheme="FP8", ignore=["lm_head"])

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-W8A8-FP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

We have successfully created an `fp8` model!

### 4) Evaluate Accuracy

With the model created, we can now load and run in vLLM (after installing).

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-W8A8-FP8")
```

We can evaluate accuracy with `lm_eval` (`pip install lm_eval==v0.4.3`):
> Note: quantized models can be sensitive to the presence of the `bos` token. `lm_eval` does not add a `bos` token by default, so make sure to include the `add_bos_token=True` argument when running your evaluations.

Run the following to test accuracy on GSM-8K:

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W8A8-FP8",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

We can see the resulting scores look good!

```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.776|±  |0.0264|
|     |       |strict-match    |     5|exact_match|↑  |0.776|±  |0.0264|
```

### Questions or Feature Request?

Please open up an issue on `vllm-project/llm-compressor`
