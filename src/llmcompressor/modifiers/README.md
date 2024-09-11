# Modifiers Overview

A `Modifier` in `llm-compressor` is an algorithm that can be applied to a model to change 
its state in some way. Some modifiers can be applied during one-shot, while others 
are relevant only during training. Below is a summary of the key modifiers available.

## Pruning Modifiers

Modifiers that introduce sparsity into a model

### [SparseGPT](./obcq/base.py)
One-shot algorithm that uses calibration data to introduce unstructured or structured 
sparsity into weights. Implementation based on [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774). A small amount of calibration data is used 
to calculate a Hessian for each layers input activations, this Hessian is then used to 
solve a regression problem that minimizes the error introduced by a target sparsity. This algorithm 
has a good amount of memory overhead introduced by storing the Hessians.

### [WANDA](./pruning/wanda/base.py)
One-shot algorithm that uses calibration data to introduce unstructured or structured sparsity. Implementation is
based on [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/pdf/2306.11695).
Calibration data is used to calculate the magnitude of input activations for each layer, and weights 
are pruned based on this magnitude combined with their distance from 0. This requires less 
memory overhead and computation than SparseGPT, but reduces accuracy in many cases.

### [Magnitude Pruning](./pruning/magnitude/base.py)
Naive one-shot pruning algorithm that does not require any calibration data. Weights are 
pruned based solely on their distance from 0 up to the target sparsity.

## Quantization Modifiers

Modifiers that quantize weights or activations of a model

### [Basic Quantization](./quantization/quantization/base.py)

### [GPTQ](./quantization/gptq/base.py)
[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/pdf/2210.17323)

## "Helper" Modifiers

These modifiers do not introduce sparsity or quantization themselves, but are used 
in conjunction with one of the above modifiers to improve their accuracy.

### [SmoothQuant](./smoothquant/base.py)
[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/pdf/2211.10438)

### [Logarithmic Equalization](./logarithmic_equalization/base.py)

### [Constant Pruning](./pruning/constant/base.py)

### [Distillation](./distillation/output/base.py)