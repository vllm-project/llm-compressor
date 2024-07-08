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

While weight-only quanitzation does not change the latency of the tensor-core operations (since the compute still runs at `bf/fp16`), it can reduce the latency of moving the weights from DRAM to SRAM with "fused" inference kernels that upconvert the weights to `fp16`after moving them into SRAM (thereby reducing the total amount of data movement between DRAM and SRAM). LLM Inference Serving is usually dominated by batch size < 64 "decode" operations, which are "memory bandwidth bound", meaning we can speed up the `Linear` matmuls with weight-only quantization.

### Accelerating Inference Serving in vLLM with `Marlin`

[`Marlin`](https://neuralmagic.com/blog/pushing-the-boundaries-of-mixed-precision-llm-inference-with-marlin/) is an optimized fused inference kernel for weight-only quantization, supporting `int4`, `int8`, and `fp8` weights with `fp16` and `bf16` activations. vLLM uses `Marlin` when executing inference for weight-only quantized models created via `llm-compressor`.

vLLM achieves strong end-to-end speedups from activation quantization on Nvidia A10G GPUs with `Meta-Llama-3-8B-Instruct` running the `sharegpt` online serving benchmark with 1 query per second:

| Weight Precision  | Activation Precision  | Time Per Output Token (ms)    | Speedup vs `fp16` |
|-                  |-                      |-                              | -                 |
|`fp16`             | `fp16`                | 42.52                         | 1.0x              |
|`fp8`              | `fp16`                | 22.95                         | 1.9x              |
|`int8`             | `fp16`                | 26.34                         | 1.6x              |
|`int4-g128`        | `fp16`                | 15.46                         | 2.8x              |

> Performance results computed as of `vllm==v0.5.1` via [online serving performance benchmark](../../examples/benchmarking/online_serving)

### Examples
- [`int4` weight-only quantization with `Meta-Llama-3-8B-Instruct`](../../examples/quantization_w4a16)

## Weights And Activation Quantization

With weights and activation quantization, we quantize both the weights and activations to lower precision (typically to `int8` or `fp8`). As a result, at inference time, we can use lower precision tensor cores to accelerate computation. Lower precision tensor cores have more TFLOPS (floating-point operations per second) available:

| GPU               | `fp16`            | `int8`            | `fp8`             |
| -                 | -                 | -                 | -                 |
| `A10G`            | 125 TFLOPS        | 250 TOPS          | Not supported     |
| `A100 SXM`        | 312 TFLOPS        | 624 TOPS          | Not supported     |
| `H100 SXM`        | 990 TFLOPS        | 1979 TOPS         | 1979 TFLOPS       |

>   - [`A10G` datasheet](https://www.nvidia.com/en-us/data-center/products/a10-gpu/)
>   - [`A100-80GB-SXM5` datasheet](https://www.nvidia.com/en-us/data-center/a100/)
>   - [`H100-80GV-SXM5` datasheet](https://www.nvidia.com/en-us/data-center/h100/)
 
As a result, activation quantization is able to accelerate both "memory bandwidth bound" and "compute bound" operations.

### Accelerating Offline Batch Processing in vLLM

vLLM supports activation quantization acceleration using custom Cutlass-based inference kernels for models created via `llm-compressor`.

Let's take a look at the end-to-end performance gains on an Nvidia A10G GPU with `Meta-Llama-3-8B-Instruct` running an offline throughput benchmark:

| Weight Precision  | Activation Precision  | Generation Throughtput         | Speedup vs `fp16` |
|-                  |-                      |-                               | -                 |
|`fp16`             | `fp16`                | 488 tok/sec                    | 1.0x              |
|`int8`             | `int8`                | 977 tok/sec                    | 2.0x              |

> Performance results computed as of `vllm==v0.5.1` via [offline performance benchmark](../../examples/benchmarking/offline_batch/)

### Examples
- [`w8a8 int8` quantization with `Meta-Llama-3-8B-Instruct`](../../examples/quantization_w8a8_int8)
- [`w8a8 fp8` quantization with `Meta-Llama-3-8B-Instruct`](../../examples/quantization_w8a8_fp8)


## Other Resources

- Horace He's blog [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) for more conceptual background on compute vs bandwidth-bound operations
- Neural Magic's blog [Pushing the Boundaries of Mixed-Precision LLM Inference With Marlin](https://neuralmagic.com/blog/pushing-the-boundaries-of-mixed-precision-llm-inference-with-marlin/) for more details on how Marlin works
