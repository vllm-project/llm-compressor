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
 
In summary, AutoRound demonstrates leading or on-par performance at 4-bit precision, with clear advantages for sub-4-bit, as reported in **SignRoundV1** ([paper](https://arxiv.org/pdf/2309.05516)), **SignRoundV2** ([paper](http://arxiv.org/abs/2512.04746)) and the **Intel Low-Bit Open LLM Leaderboard** ([link](https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard)),
 
**INT4 for Large Models (≈30B and above)**
AutoRound achieves performance comparable to other PTQ methods, as the accuracy drop for these large models is generally minimal.
 
**INT4 for Small-to-Medium LLMs**
AutoRound is likely to deliver higher accuracy than existing PTQ methods, making it particularly effective for smaller models. See SignRoundV1 And Low Bit Open LLM Leaderboard for accuracy data.
 
**Sub-4-Bit Quantization (INT2/INT3)**
As the bit-width decreases, AutoRound shows increasing benefits, achieving 10–20% absolute accuracy improvements over PTQ methods, while matching QAT performance at 1–2 orders of magnitude lower tuning cost. See SignRound V2 for details.
 
**New Data Types (MXFP4 / NVFP4)**
For emerging floating-point formats, AutoRound consistently outperforms RTN in accuracy, demonstrating strong forward compatibility with evolving quantization standards. See SignRound V2 for details.

### Key Parameters
- `scheme`: Quantization scheme (e.g., `W4A16`, `W8A16`, more schemes will be supported soon)
- `iters`: Number of tuning iterations per block. Default: 200
- `batch_size`: Batch size for calibration. Default: 8
- `lr`: Learning rate for tuning. If `None`, auto-set to `1.0/iters`. Default: `None`
- `NUM_CALIBRATION_SAMPLES`: Number of calibration samples. Default: 128
- `MAX_SEQUENCE_LENGTH`: Sequence length of calibration samples. Default: 2048


### Quantization Configurations

The accuracy of the quantized model is configured by tuning-related parameters. AutoRound provides four recommended configurations to balance accuracy and quantization speed:

| Mode    | Batch Size | Iterations | Sequence Length | Calibration Samples | Learning Rate | Quantization Speed | Memory Usage | Accuracy   |
|---------|------------|------------|-----------------|---------------------|---------------|--------------------|--------------|------------|
|`default`| 8          | 200        | 2048            | 128                 | Auto          | 🚀🚀              | 🟡 Medium    | 🎯🎯 Good |
|`best`   | 8          | 1000       | 2048            | 512                 | Auto          | 🚀                | 🔴 High      | 🏆 Best    |
|`light`  | 8          | 50         | 2048            | 128                 | 5e-3          | 🚀🚀🚀           | 🟡 Medium    | 🎯🎯 (slight drop in some cases) |
|`fast`   | 4          | 200        | 512             | 128                 | Auto          | 🚀🚀🚀           | 🟢 Low       | 🎯         |

> [!TIP]
> - Use `best` for production models where accuracy is critical
> - Use `light` for rapid iteration during development (2-3× speedup)
> - Use `fast` when GPU memory is limited or for quick evaluation
> - The `default` recipe provides a good balance for most use cases

> [!NOTE]
> These configurations are based on our experiments and may vary depending on the model architecture.


### Support Matrix
| Scheme              | Examples                                                                  | Note                                  |
| ------------------- | ------------------------------------------------------------------------- | ------------------------------------- |
| `wNa16`             | [llama3_example](./quantization_w4a16/llama3_example.py)                  |                                       |
| `wNa16`             | [qwen3_example](./quantization_w4a16/qwen3_example.py)                    | Multiple cards for `Qwen3-235B-A22B`  |
| `wNa16` + `FP8KV`   | [llama3_example](./quantization_kv_cache/llama3_example.py)               |                                       |
| `W8A8-FP8` Static   | [llama4_example](./quantization_w8a8_fp8/llama4_static_quant_example.py) |                                       |
| `W8A8-FP8` Dynamic  | [llama4_example](./quantization_w8a8_fp8/llama4_dynamic_quant_example.py)  |                                       |
| `NVFP4`  | [llama3.1_example](./quantization_w4a4_fp4/llama3.1_example.py)  |                                       |


> [!NOTE]
> More quantization schemes (e.g., `MXFP4`) are actively being developed. Stay tuned for updates!


### Known Issues
Currently, `llm-compressor` supports applying AutoRound only on the WNA16, NVFP4, and W8A8-FP8 quantization schemes. Support for additional schemes is planned. You can follow progress in the [RFC](https://github.com/vllm-project/llm-compressor/issues/1968).

### Questions or Feature Requests?

Please open up an issue on [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) or [intel/auto-round](https://github.com/intel/auto-round).
