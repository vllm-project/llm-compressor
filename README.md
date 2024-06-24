# LLM Compressor
Easy to use model compression library that supports a growing list of formats including
int and float quantization, activation quantization, and 2:4 sparsity.

## Overview
A library for compressing large language models utilizing the latest techniques and research for both training aware and post training techniques.
llm-compressor is designed to be flexible and easy to use on top of PyTorch and HuggingFace Transformers, allowing for quick experimentation.
Compression algorithms are implemented as `Modifiers` which can be applied to create optimized models for inference.
The library also emphasises support for deployment on vLLM through export to the compressed-tensors format.

## Installation

### Pip

Coming Soon!

### From Source
llm-compressor can be installed from the source code via a git clone and local pip install.

```bash
git clone https://github.com/vllm-project/llm-compressor.git
pip install -e llm-compressor
```

## Quick Tour
The following snippet is a minimal example for compression and inference of a Llama model.
The 1.1B model may be swapped to another model in the HuggingFace Hub or custom local model.
This example uses 4-bit weight quantization, however the `scheme` may be changed to
target different quantization scenarios.


### Compression
Compression is easily applied by selecting an algorithm (GPTQ) and calling the `oneshot` API.

```python
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization.gptq import GPTQModifier

# sets parameters for the GPTQ algorithms - target Linear layer weights at 4 bits
gptq = GPTQModifier(scheme="W4A16", targets="Linear")

oneshot(
    model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",  # sample model
    dataset="open_platypus",  # calibration dataset, swap with yours
    recipe=gptq,
    save_compressed=True,
    output_dir="llama-compressed-quickstart",
    overwrite_output_dir=True,
    max_seq_length=256,
    num_calibration_samples=256,
)
```

### Inference with vLLM
To run inference with vLLM, first install vLLM from pip `pip install vllm`.

```python
from vllm import LLM

model = LLM("llama-compressed-quickstart")
output = model.genetate("I love 4 bit models because")
```


## Learn More
The llm-compressor library provides a rich feature-set for model compression below are examples
and documentation of a few key flows.  If you have any questions or requests
open an [issue](https://github.com/vllm-project/llm-compressor/issues) and we will add an example or documentation.

* One-Shot API Documentation (Coming Soon!)
* Modifiers Documentation (Coming Soon!)
* [One-Shot Quantization](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization/llama7b_one_shot_quantization.md)
* [Creating a Sparse-Quantized Llama-7b model](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization/llama7b_one_shot_quantization.md)

## Contribute
We appreciate contributions to the code, examples, integrations, and documentation as well as bug reports and feature requests!
[Learn how here](https://github.com/vllm-project/llm-compressor/blob/main/CONTRIBUTING.md).