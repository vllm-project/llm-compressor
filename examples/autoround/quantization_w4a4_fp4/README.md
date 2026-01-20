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
python3 llama3.1_example.py
```

The resulting model `Meta-Llama-3.1-8B-Instruct-NVFP4-AutoRound` is ready to be loaded into vLLM.

### Evaluate Accuracy

With the model created, we can now load and run in vLLM (after installing).

```python
from vllm import LLM
model = LLM("./Meta-Llama-3.1-8B-Instruct-NVFP4-AutoRound")
```

We can evaluate accuracy with `lm_eval` (`pip install lm-eval==0.4.9.1`):
> Note: quantized models can be sensitive to the presence of the `bos` token. `lm_eval` does not add a `bos` token by default, so make sure to include the `add_bos_token=True` argument when running your evaluations.

Run the following to test accuracy on GSM-8K:

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3.1-8B-Instruct-NVFP4-AutoRound",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size 'auto'
```

#### meta-llama/Meta-Llama-3.1-8B-Instruct
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7710|±  |0.0116|
|     |       |strict-match    |     5|exact_match|↑  |0.7043|±  |0.0126|

#### Meta-Llama-3.1-8B-Instruct-NVFP4 (QuantizationModifier)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7248|±  |0.0123|
|     |       |strict-match    |     5|exact_match|↑  |0.6611|±  |0.0130|


#### Meta-Llama-3.1-8B-Instruct-NVFP4-AutoRound (AutoRoundModifier, iters=0)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7362|±  |0.0121|
|     |       |strict-match    |     5|exact_match|↑  |0.6702|±  |0.0129|

#### Meta-Llama-3.1-8B-Instruct-NVFP4-AutoRound (AutoRoundModifier, iters=200)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7210|±  |0.0124|
|     |       |strict-match    |     5|exact_match|↑  |0.6945|±  |0.0127|

> Note: quantized model accuracy may vary slightly due to nondeterminism.

### Questions or Feature Request?

Please open up an issue on [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) or [intel/auto-round](https://github.com/intel/auto-round).
