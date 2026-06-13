# Quantizing the Input Embedding

`llm-compressor` can quantize a model's **input embedding table** (the vocab
lookup) to weight-only `intN` (`WNA16`), in addition to the linear layers. vLLM
loads these checkpoints and runs an efficient fused gather + dequant for the
looked-up rows, so the packed table is never densified.

Embedding quantization is most useful for **large-vocabulary models**, where the
embedding table is a meaningful fraction of memory (e.g. a 150k-vocab model has a
~1 GB embedding table in fp16). It is **weight-only and data-free** -- no
calibration dataset is required -- and its accuracy impact is typically
negligible.

## Installation

```bash
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
pip install -e .
```

## Quickstart

```bash
python3 llama3_example.py
```

The resulting model `Meta-Llama-3-8B-Instruct-embedding-W4A16-G64` is ready to be
loaded into vLLM.

## Code Walkthrough

### 1) Load the model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

### 2) Apply embedding quantization (data-free)

Target the `Embedding` module **by class name** rather than by a layer-name
regex. Class-based matching is portable across architectures and does not depend
on the module's prefix, so the same recipe works for `model.embed_tokens`,
`gpt_neox.embed_in`, and so on:

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    config_groups={
        "embedding": {
            "targets": ["Embedding"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 64,
            },
        }
    }
)

# Weight-only -> data-free; no calibration dataset needed.
oneshot(model=model, recipe=recipe)
```

Use `"strategy": "channel"` (and drop `group_size`) for per-row channel scales,
or `"num_bits": 8` for a more conservative, effectively lossless setting.

### 3) Save the compressed model

```python
SAVE_DIR = "Meta-Llama-3-8B-Instruct-embedding-W4A16-G64"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

## Composing with weight quantization

Embedding quantization can be combined with linear-weight quantization by adding
the linear modifier alongside it, e.g.:

```python
from llmcompressor.modifiers.quantization import GPTQModifier

recipe = [
    GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
    QuantizationModifier(config_groups={"embedding": embedding_config}), # where embedding_config is defined as above
]
```

## Accuracy

Quantizing only the input embedding is near-lossless. On `pythia-1.4b`
(`lm-eval`, `wikitext` word perplexity / `arc_easy` acc):

| scheme | wikitext ppl | arc_easy acc |
| --- | --- | --- |
| baseline (fp16) | 14.733 | 0.6048 |
| embedding W8 channel | 14.732 | 0.6052 |
| embedding W4 group-64 | 14.752 | 0.6061 |
