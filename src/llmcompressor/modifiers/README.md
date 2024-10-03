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
One-shot algorithm that quantizes weights, input activations and/or output activations by 
calculating a range from weights or calibration data. All data is quantized to the closest 
bin using a scale and (optional) zero point. This basic quantization algorithm is 
suitable for FP8 quantization. A variety of quantization schemes are supported via the 
[compressed-tensors](https://github.com/neuralmagic/compressed-tensors) library. 

### [GPTQ](./quantization/gptq/base.py)
One-shot algorithm that uses calibration data to select the ideal bin for weight quantization. 
This algorithm is applied on top of the basic quantization algorithm, and affects weights only.
The implementation is based on [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/pdf/2210.17323). The algorithm is very similar to SparseGPT: A small amount of calibration data is used 
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

### [Constant Pruning](./pruning/constant/base.py)
One-shot pruning algorithms often introduce accuracy degradation that can be recovered with finetuning. This 
modifier ensures that the sparsity mask of the model is maintained during finetuning, allowing a sparse 
model to recover accuracy while maintaining its sparsity structure. It is intended to be used after a pruning modifier
such as `SparseGPT` or `WANDA` has already been applied.

### [Distillation](./distillation/output/base.py)
To better recover accuracy of sparse models during finetuning, we can also use a teacher model of the same architecture
to influence the loss. This modifier is intended to be used in conjunction with `ConstantPruning` modifier on a 
pruned model, with the dense version of the model being used as the teacher. Both output distillation loss and 
layer-by-layer distillation loss are supported. The layer-by-layer implementation follows the Square Head distillation 
algorithm presented in [Sparse Fine-tuning for Inference Acceleration of Large Language Models](https://arxiv.org/pdf/2310.06927).