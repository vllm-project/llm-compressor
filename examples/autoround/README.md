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

## Detailed Usage
- iters
- scheme
- enable_torch_compile
- batch_size
- num_calibration_samples
- seqlen

### Tips


Could you please provide more details about the evaluation setup? If the evaluation is not very heavy, we could rerun a subset of experiments to double-check whether the issue comes from our upstream pipeline or whether AutoRound (AR) indeed underperforms in this case.
Overall, AutoRound shows advantages in several scenarios; however, as model size increases and bit-width goes up, this advantage tends to diminish.
INT4 for small- to medium-sized LLMs.
See Paper 1 for a fair comparison where GPTQ has act_order enabled. At W4G128, AR wins in 3 out of 4 cases. Across 2–4 bits, AR wins in 14 out of 16 cases. For a broader (though slightly unfair as AR typically use best recipe) comparison, please refer to the Intel low-bit Open LLM leaderboard , where AR wins in most cases, https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard.
Sub-4-bit settings (INT2 / INT3).
As shown in Papers 1 and 2, AR demonstrates a clear advantage at 2 bits, typically achieving a 10–20% absolute accuracy improvement.
New data types (MXFP4 / NVFP4).
As discussed in Paper 2, due to the lack of GPTQ baselines, we cannot provide a direct comparison. Nevertheless, the results show that AR consistently improves accuracy for these data types.

### Support Matrix
| Scheme  | Examples | Note |
| ------- | -------- | ---- |
| `wNa16` |          |      |
| `FP8`   |          |      |
| `NVFP4` |          |      |



### Known Issues
Currently, `llm-compressor` supports applying AutoRound only on the `wNa16` quantization schemes. Support for additional schemes is planned. You can follow progress in the [RFC](https://github.com/vllm-project/llm-compressor/issues/1968).

### Questions or Feature Request?

Please open up an issue on [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) or [intel/auto-round](https://github.com/intel/auto-round).
