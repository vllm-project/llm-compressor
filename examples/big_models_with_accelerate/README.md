# Quantizing Big Models with `accelerate`

`llmcompressor` integrates natively wiht `accelerate` to support quantizing large models.

## Overview

[`accelerate`]((https://huggingface.co/docs/accelerate/en/index)) is a highly useful library in the Hugging Face ecosystem that supports for working with large models, including:
- Offloading parameters to CPU and disk
- Sharding models across multiple GPUs with pipeline-parallelism

As a result, `accelerate` enables `llmcompressor` to easily support quantizing models that cannot fit onto a single GPU (such as Llama 70B and 405B)!

When working with `accelerate`, it is important to keep in mind that CPU offloading and naive pipeline-parallelism will slow down forward passes through the model. As a result, we need to take care to ensure that the quantization methods used fit well with the offloading scheme. As a general rule of thumbs:
- CPU offloading can be used with data-free quantization methods (e.g. PTQ with `FP8_DYNAMIC`)
- Multi-GPU can can be used with calibration data-based methods, but be careful with

In this guide, we will show examples of how to:
- Quantize Llama-70B to W8A8 (FP8) using PTQ on a single GPU (with CPU offloading)
- Quantize Llama-70B to W8A8 (INT8) using GPTQ and SmoothQuant on 8 H100s

## Examples

### Install

To get started, install `llmcompressor`:

```bash
pip install llmcompressor==0.1.0
```

### CPU Offloading with `FP8` W8A8 Quantization

CPU offloading is extremely slow. As a result, we recommend using this feature only with ***data-free quantization methods***. An example data free quantization method is `FP8_DYNAMIC` quantization, which uses PTQ to statically quantize the weights with dynamic activation quantization and therefore does not require calibration data.

To enable CPU offloading, we simply need to adjust the `device_map` used in `from_pretrained` with `SparseAutoModel`:

```python
from llmcompressor.transformers import SparseAutoModelForCausalLM
MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"

# device_map="auto" will offload any parameters that cannot fit into CPU RAM.
model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto")
```

If there is not enough GPU memory to hold the model, `device_map="auto"` will offload the weights to the CPU. `llmcompressor` is designed to work properly with `accelerate`, so calls to `oneshot` will respect the CPU offloading.

#### Working Example

`cpu_offloading_fp8.py` demonstrates quantizing the weights and activations of `meta-llama/Meta-Llama-3.1-70B-Instruct` to `fp8` on a single GPU:

```python
export CUDA_VISIBLE_DEVICES=0
python3 cpu_offloading_fp8.py
```

The resulting model `./Meta-Llama-3-70B-Instruct-FP8-Dynamic` is quantized and ready to run with `vllm`!

### Multi-GPU for `INT8` W8A8 Quantization with `GPTQ` and `SmoothQuant`

For quantization methods that require calibration data (e.g. `GPTQ` and `SmoothQuant`), CPU offloading is typically too slow. For these methods, `llmcompressor` can use `accelerate` multi-GPU to quantize models that are larger than a single GPU can fit.

To enable multi-GPU, we simply need to adjust the `device_map` used in `from_pretrained` with `SparseAutoModel`:

```python
from llmcompressor.transformers import SparseAutoModelForCausalLM
MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"

# device_map="auto" shards the model over all visible GPUs.
model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto")
```

`llmcompressor` is designed to work properly with `accelerate`, so calls to `oneshot` will work in a multi-GPU setup.


#### Working Example

`multi_gpu_int8.py` demonstrates quantizing the weights and activations of `meta-llama/Meta-Llama-3.1-70B-Instruct` to `int8` on 8 A/H100 GPUs:

```python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 multi_gpu_int8.py
```

The resulting model `./Meta-Llama-3-70B-Instruct-INT8-Dynamic` is quantized and ready to run with `vllm`!