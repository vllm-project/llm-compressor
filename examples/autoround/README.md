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



## When to Use AutoRound

AutoRound demonstrates clear advantages in specific quantization scenarios:

**INT4 for Small-to-Medium LLMs**: At W4G128 configuration, AutoRound outperforms GPTQ (with `act_order` enabled) in 3 out of 4 cases. Across 2–4 bit ranges, AutoRound wins in 14 out of 16 benchmark comparisons. See the [Intel Low-Bit Open LLM Leaderboard](https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard) for comprehensive results.

**Sub-4-Bit Quantization (INT2/INT3)**: AutoRound excels at aggressive compression, achieving 10–20% absolute accuracy improvements over alternatives at 2-bit quantization.

**New Data Types (MXFP4/NVFP4)**: For emerging floating-point formats, AutoRound consistently delivers accuracy improvements, positioning it as forward-compatible with evolving quantization standards.

**Note**: AutoRound's advantages tend to diminish as model size increases and bit-width goes higher (approaching 8-bit), where quantization challenges are already less severe.

### Key Parameters

- iters
- scheme
- enable_torch_compile
- batch_size
- seqlen
- lr
- MAX_SEQUENCE_LENGTH
- NUM_CALIBRATION_SAMPLES

- `scheme`: Quantization scheme (`W4A16`, `W8A16`)
- `iters`: Number of tuning iterations per decoding layer
- `batch_size`: Batch size for calibration
- `enable_torch_compile`: Enable PyTorch compilation for faster execution
- `lr`: Learning rate for tuning (if None, auto-set to 1.0/iters)
- `NUM_CALIBRATION_SAMPLES`: Number of calibration samples
- `MAX_SEQUENCE_LENGTH`: Sequence length of calibration samples

### Quantization Recipes

AutoRound provides four built-in recipes to balance accuracy and quantization speed:

| Recipe    | Batch Size | Iterations | Sequence Length | Calibration Samples | Learning Rate | Use Case |
|-----------|------------|------------|-----------------|---------------------|---------------|----------|
| `default` | 8          | 200        | 2048            | 128                 | Auto          | Balanced accuracy and speed |
| `best`    | 8          | 1000       | 2048            | 512                 | Auto          | Highest accuracy, 4-5× slower |
| `light`   | 8          | 50         | 2048            | 128                 | 5e-3          | Fast quantization, slight accuracy drop at W4G128 |
| `fast`    | 4          | 200        | 512             | 128                 | Auto          | Memory-constrained environments |

> [!TIP]
> - Use `best` for production models where accuracy is critical
> - Use `light` for rapid iteration during development (2-3× speedup)
> - Use `fast` when GPU memory is limited or for quick evaluation
> - The `default` recipe provides a good balance for most use cases



### Support Matrix (WIP)
| Scheme  | Examples | Note |
| ------- | -------- | ---- |
| `wNa16` |          |      |
| `FP8`   |          |      |
| `NVFP4` |          |      |



### Known Issues
Currently, `llm-compressor` supports applying AutoRound only on the `wNa16` quantization schemes. Support for additional schemes is planned. You can follow progress in the [RFC](https://github.com/vllm-project/llm-compressor/issues/1968).

### Questions or Feature Request?

Please open up an issue on [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) or [intel/auto-round](https://github.com/intel/auto-round).
