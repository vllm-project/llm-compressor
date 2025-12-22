# `AutoRound` Quantization

`llm-compressor` supports [AutoRound](https://aclanthology.org/2024.findings-emnlp.662.pdf), an advanced quantization technique that delivers **high-accuracy**, **low-bit quantization**. The quantized results are fully compatible with `compressed-tensors` and can be served directly with vLLM.

AutoRound introduces three trainable parameters (V, α, and β) to optimize rounding values and clipping ranges during quantization. The method processes each decoder layer sequentially, using block-wise output reconstruction error as the training objective to fine-tune these parameters. This approach combines the efficiency of post-training quantization with the adaptability of parameter tuning, delivering robust compression for large language models while maintaining strong performance.

## Installation

To get started, install:

```bash
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

## Quickstart

The example includes an end-to-end script for applying the AutoRound quantization algorithm.

```bash
python3 llama3_example.py
```

The resulting model `Meta-Llama-3-8B-Instruct-W4A16-G128-AutoRound` is ready to be loaded into vLLM.

## Code Walkthrough

Now, we will step through the code in the example. There are four steps:
1) Load model
2) Prepare calibration data
3) Apply quantization
4) Evaluate accuracy in vLLM

### 1) Load Model

Load the model using `AutoModelForCausalLM` for handling quantized saving and loading. 

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2) Prepare Calibration Data

When quantizing model weights with AutoRound, you’ll need a small set of sample data to run the algorithm. By default, we are using [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) as our calibration dataset.
Recommended starting points:
- 128 samples — typically sufficient for stable calibration (increase if accuracy degrades).
- 2048 sequence length — a good baseline for most LLMs.
- 200 tuning steps — usually enough to converge (increase if accuracy drops).

```python
# Select calibration dataset.
from auto_round.calib_dataset import get_dataset

NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048

# Get aligned calibration dataset.
ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)
```

### 3) Apply Quantization

With the dataset ready, we will now apply AutoRound quantization to the model.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier

# Configure the quantization algorithm to run.
recipe = AutoRoundModifier(
    targets="Linear", scheme="W4A16", ignore=["lm_head"], iters=200
)

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    # disable shuffling to get slightly better mmlu score
    shuffle_calibration_samples=False,
)


# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-G128-AutoRound"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

We have successfully created an `int4` model!

### 4) Evaluate Accuracy

With the model created, we can now load and run in vLLM (after installing).

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-W4A16-G128-AutoRound")
```

We can evaluate accuracy with `lm_eval` (`pip install lm-eval==0.4.9.1`):
> Note: quantized models can be sensitive to the presence of the `bos` token. `lm_eval` does not add a `bos` token by default, so make sure to include the `add_bos_token=True` argument when running your evaluations.

Run the following to test accuracy on GSM-8K:

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W4A16-G128-AutoRound",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 1000 \
  --batch_size 'auto'
```

We can see the resulting scores look good!

```bash
| Tasks | Version | Filter           | n-shot | Metric      |     | Value |     | Stderr |
| ----- | ------: | ---------------- | -----: | ----------- | --- | ----: | --- | -----: |
| gsm8k |       3 | flexible-extract |      5 | exact_match | ↑   | 0.737 | ±   | 0.0139 |
|       |         | strict-match     |      5 | exact_match | ↑   | 0.736 | ±   | 0.0139 |
```
> Note: quantized model accuracy may vary slightly due to nondeterminism.

### Known Issues
Currently, `llm-compressor` supports applying AutoRound only on the `wNa16` quantization schemes. Support for additional schemes is planned. You can follow progress in the [RFC](https://github.com/vllm-project/llm-compressor/issues/1968).

### Questions or Feature Request?

Please open up an issue on [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) or [intel/auto-round](https://github.com/intel/auto-round).
