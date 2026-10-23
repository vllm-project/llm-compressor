# Modifiers Overview

A `Modifier` in `llm-compressor` is an algorithm that can be applied to a model to change 
its state in some way. Some modifiers can be applied during one-shot, while others 
are relevant only during training. Below is a summary of the key modifiers available.

## Pruning Modifiers

Modifiers that prune a model

### [REAP](./pruning/reap/base.py)
One-shot algorithm for Mixture-of-Experts models that uses calibration data to score
experts by router-weighted activation saliency, then structurally removes the
least-salient experts from each MoE layer.

## Quantization Modifiers

Modifiers that quantize weights or activations of a model

### [Basic Quantization](./quantization/quantization/base.py)
One-shot algorithm that quantizes weights, input activations and/or output activations by 
calculating a range from weights or calibration data. All data is quantized to the closest 
bin using a scale and (optional) zero point. This basic quantization algorithm is 
suitable for FP8 quantization. A variety of quantization schemes are supported via the 
[compressed-tensors](https://github.com/vllm-project/compressed-tensors) library. 

### [GPTQ](./gptq/base.py)
One-shot algorithm that uses calibration data to select the ideal bin for weight quantization. 
This algorithm is applied on top of the basic quantization algorithm, and affects weights only.
The implementation is based on [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/pdf/2210.17323). The algorithm works as follows: A small amount of calibration data is used 
to calculate a Hessian for each layers input activations, this Hessian is then used to 
solve a regression problem that minimizes the error introduced by a given quantization configuration. This algorithm 
has a good amount of memory overhead introduced by storing the Hessians.

## "Helper" Modifiers

These modifiers do not introduce sparsity or quantization themselves, but are used 
in conjunction with one of the above modifiers to improve their accuracy.

### [SmoothQuant](./smoothquant/base.py)
The modifier is intended to be used prior to a `QuantizationModifier` or `GPTQModifier`. Its purpose is 
to make input activations easier to quantize by smoothing away outliers in the inputs, and applying the inverse 
smoothing operation to the following weights. This makes weights slightly harder to quantize, but the inputs much
easier to quantize. The implementation is based on [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/pdf/2211.10438) and requires calibration data. 

### [Logarithmic Equalization](./logarithmic_equalization/base.py)
Very similar to `SmoothQuantModifier`, but applies smoothing on an inverse log scale 
rather than the linear smoothing done by SmoothQuant. The implementation is based on 
[FPTQ: Fine-grained Post-Training Quantization for Large Language Models](https://arxiv.org/pdf/2308.15987)
