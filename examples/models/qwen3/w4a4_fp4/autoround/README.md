# `AutoRound` Quantization

`llm-compressor` supports [AutoRound](https://aclanthology.org/2024.findings-emnlp.662.pdf), an advanced quantization technique that delivers **high-accuracy**, **low-bit quantization**. The quantized results are fully compatible with `compressed-tensors` and can be served directly with vLLM.

AutoRound introduces three trainable parameters (V, α, and β) to optimize rounding values and clipping ranges during quantization. The method processes each decoder layer sequentially, using block-wise output reconstruction error as the training objective to fine-tune these parameters. This approach combines the efficiency of post-training quantization with the adaptability of parameter tuning, delivering robust compression for large language models while maintaining strong performance.

## Qwen3-VL Example

```bash
python3 qwen3_vl_example.py
```

The resulting model `Qwen3-VL-8B-Instruct-NVFP4-AutoRound` is ready to be loaded into vLLM.

### Evaluate Accuracy

Run the following to test accuracy on GSM-8K and ChartQA:

```bash
lm_eval --model vllm-vlm \
  --model_args pretrained="./Qwen3-VL-8B-Instruct-NVFP4-AutoRound",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size 'auto'

lm_eval --model vllm-vlm \
  --model_args pretrained="./Qwen3-VL-8B-Instruct-NVFP4-AutoRound",add_bos_token=true \
  --tasks chartqa \
  --batch_size 'auto' \
  --apply_chat_template
```

#### Qwen/Qwen3-VL-8B-Instruct (Baseline)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8628|±  |0.0095|
|     |       |strict-match    |     5|exact_match|↑  |0.8453|±  |0.0100|

| Tasks |Version|Filter|n-shot|     Metric      |   |Value |   |Stderr|
|-------|------:|------|-----:|-----------------|---|-----:|---|-----:|
|chartqa|      0|none  |     0|anywhere_accuracy|↑  |0.7908|±  |0.0081|
|       |       |none  |     0|exact_match      |↑  |0.5592|±  |0.0099|
|       |       |none  |     0|relaxed_accuracy |↑  |0.7696|±  |0.0084|

#### Qwen3-VL-8B-Instruct-NVFP4-AutoRound (AutoRoundModifier, iters=200)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8415|±  |0.0101|
|     |       |strict-match    |     5|exact_match|↑  |0.8408|±  |0.0101|

| Tasks |Version|Filter|n-shot|     Metric      |   |Value |   |Stderr|
|-------|------:|------|-----:|-----------------|---|-----:|---|-----:|
|chartqa|      0|none  |     0|anywhere_accuracy|↑  |0.8220|±  |0.0077|
|       |       |none  |     0|exact_match      |↑  |0.5748|±  |0.0099|
|       |       |none  |     0|relaxed_accuracy |↑  |0.8044|±  |0.0079|

> Note: quantized model accuracy may vary slightly due to nondeterminism.

### Questions or Feature Request?

Please open up an issue on [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) or [intel/auto-round](https://github.com/intel/auto-round).
