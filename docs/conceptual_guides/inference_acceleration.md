# Inference Acceleration with Quantization

There are two "types" of quantization, each of which can accelerate inference:
* Weight-Only Quantization
* Weight and Activation Quantization

## Weight-Only Quantization

With weight-only quantization, weights are quantized to low precision (typically `int8`, `fp8`, or `int4`) while activations remain at higher precision `fp16` or `bf16`. To perform matrix multiplication, we upconvert each weight to`fp16` before computing `A*B`.

### How Can We Speed Up Weight-Only Quantization?

Roughly speaking, the time required to execute a matrix multiplication on a GPU equals the sum of:
* Latency of moving the weights from main memory (DRAM) to the compute (SRAM)
* Latency of the tensor-core compute operations

Weight-only quanitzation does nnot impact latency of the tensor-core operations, but it can reduce the amount data moving from DRAM to SRAM with "fused" inference kernels that upconvert the weights to `fp16` after moving them into SRAM.

### Accelerating Inference Serving

Since LLM serving is usually dominated by "decode" operations, which are "memory bandwidth bound", weight-only quantization is quite useful for accelerating online-servig.

[`Marlin`](https://neuralmagic.com/blog/pushing-the-boundaries-of-mixed-precision-llm-inference-with-marlin/) is an optimized fused inference kernel for weight-only quantization, supporting `int4`, `int8`, and `fp8` weights with `fp16` and `bf16` activations. vLLM uses `Marlin` when executing inference for weight-only quantized models created via `llm-compressor`.

End-to-end speedups on for `Meta-Llama-3-8B-Instruct` on A10G with 1 QPS:
| Weight Precision  | Activation Precision  | Time Per Output Token (ms)    | Speedup vs `fp16` |
|-                  |-                      |-                              | -                 |
|`fp16`             | `fp16`                | 42.52                         | 1.0x              |
|`fp8`              | `fp16`                | 22.95                         | 1.9x              |
|`int8`             | `fp16`                | 26.34                         | 1.6x              |
|`int4-g128`        | `fp16`                | 15.46                         | 2.8x              |

> Performance results computed as of `vllm==v0.5.1` via [online serving performance benchmark](../../examples/benchmarking/online_serving)

### Examples
- [`int4` weight-only quantization with `Meta-Llama-3-8B-Instruct`](../../examples/quantization_w4a16)

## Weight and Activation Quantization

With weight and activation quantization, both the weights and activations are converted to to `int8` or `fp8`. At inference time, we can use low precision tensor cores, which have more FLOPS available:

| GPU       | `fp16`            | `int8`            | `fp8`             |
| -         | -                 | -                 | -                 |
| `A10G`    | 125 TFLOPS        | 250 TOPS          | Not supported     |
| `A100`    | 312 TFLOPS        | 624 TOPS          | Not supported     |
| `H100`    | 990 TFLOPS        | 1979 TOPS         | 1979 TFLOPS       |

>   [`A10G` datasheet](https://www.nvidia.com/en-us/data-center/products/a10-gpu/) // [`A100` datasheet](https://www.nvidia.com/en-us/data-center/a100/) // [`H100` datasheet](https://www.nvidia.com/en-us/data-center/h100/)
 
As a result, activation quantization is able to accelerate both "memory bandwidth bound" and "compute bound" operations.

### Accelerating Offline Batch Processing

With offline batch processing, we can crank-up the batch size as high as possible to maximize throughput, making offline batch processing "compute-bound". This means that activation quantization is very useful for accelerating performance.

vLLM supports activation quantization acceleration using custom Cutlass-based inference kernels for models created via `llm-compressor`.

End-to-end speedups on for `Meta-Llama-3-8B-Instruct` on A10G for offline batch processing:
| Weight Precision  | Activation Precision  | Generation Throughtput         | Speedup vs `fp16` |
|-                  |-                      |-                               | -                 |
|`fp16`             | `fp16`                | 488 tok/sec                    | 1.0x              |
|`int8`             | `int8`                | 977 tok/sec                    | 2.2x              |

> Performance results computed as of `vllm==v0.5.1` via [offline performance benchmark](../../examples/benchmarking/offline_batch/)

### Examples
- [`w8a8 int8` quantization with `Meta-Llama-3-8B-Instruct`](../../examples/quantization_w8a8_int8)
- [`w8a8 fp8` quantization with `Meta-Llama-3-8B-Instruct`](../../examples/quantization_w8a8_fp8)


## Other Resources

- Horace He's blog [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) for more conceptual background on compute vs bandwidth-bound operations
- Neural Magic's blog [Pushing the Boundaries of Mixed-Precision LLM Inference With Marlin](https://neuralmagic.com/blog/pushing-the-boundaries-of-mixed-precision-llm-inference-with-marlin/) for more details on how Marlin works
