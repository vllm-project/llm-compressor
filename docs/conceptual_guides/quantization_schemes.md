# Quantization Schemes

Quantization is a technique to reduce the computational and memory costs of running inference by representing the weights and activations with low-precision data types like 8-bit integer (`int8`) instead of the usual 16-bit floating point (`float16`).

## Theory

Performing quantization to go from `float16` to `int8` (or lower) is tricky. Only 256 values can be represented in `int8`, while `float16` can represent a very wide range of values. The idea is to find the best way to project our range [a, b] of `float32` values to the `int8` space.

Let’s consider a float x in [a, b], then we can write the following quantization scheme:

```bash
x = S * (x_q - Z)
```

where:

- `x_q` is the quantized `int8` value associated to `x`
- `S` is the scale, and is a positive `float16`.  It is used to "rescale" a distribution from the base range in `float16` to the desired width (ie 256 for `int8`).
- `Z` is called the zero-point, it is the `int8` value corresponding to the value 0 in the `float16` realm. If zero-point is ommited, we call this "symmetric" quantization because the default zero point of 0 is in the true middle of the distribution.


The quantized value x_q of x in [a, b] can be computed as follows:

```bash
x_q = round(x/S + Z)
```

And `float16` values outside of the [a, b] range are clipped to the closest representable value, so for any floating-point number x:

```bash
x_q = clip(round(x/S + Z), round(a/S + Z), round(b/S + Z))
```

## Quantization Flavors

There are several flavors of quantization.

### Static vs Dynamic

The section above described how quantization from `float16` to `int8` works, but did not explain how to compute the scales and zero points.

With weights, since the full range is known ahead of time, we can just compute the scales and zero points statically (sometimes using a more sophisticated algorithm like `GPTQ`).

With activations, however, there are two approaches:
* **Dynamic quantization**: the range for each activation is computed on the fly at runtime so that the quantization range matches exactly the current runtime range. This gives us the best possible values, but it can be a bit slower than static quantization because of the overhead introduced by computing the range each time. It is also not an option on certain hardware.

* **Static quantization**: the range for each activation is computed in advance at quantization-time.  This is typically done by passing representative "calibration" data through the model and recording the range of activation values. In practice, we run a number of forward passes on a calibration dataset is done and compute the ranges according to the observed calibration data.

In general, it is best practice to start your experiments with:
- For `fp8`, use static activation quantization
- For `int8`, use dynamic activation quantization

### Granularity

Weight and activation quantization can be performed at different levels of granularity depending on accuracy / latency tradeoffs being targeted.

#### Weight Quantization Granularity

For weight quantization, there are three "levels" (in order of increasing of granularity):
* **Per-Tensor**: one pair of quantization parameters (`S, Z`) is used per tensor
* **Per-Channel**: one pair of quantization parameters (`S, Z`) is used per element of one of the dimensions of the tensor. For instance, with a weight matrix of shape `[N,M]`, the scales are a vector of shape [`M`] scales.
* **Per-Group**: one pait of quantization parameters is (`S, Z`) is used per group of items in a tensor. For instance, with a weight matrix of shape `[N,M]` with `M=4096`, the scales are a matrix of shape `[32, M]` (note: `4096 / 128 = 32`).

Incresing quantization granularity typically helps with accuracy at the expense of less memory reduction and slower inference performance.  This is because we compute quantization ranges over smaller distributions with the trade off of needing more memory to represent them. In general, it is best practice to start your experiments with:
- For `int4` weights, use `per-group (size=128)`
- For `int8` weights, use `per-channel`
- For `fp8` weights, use `per-tensor`

#### Activation Quantization Granularity

For activation quantization, there are two "levels" (in order of increasing granularity):
* **Per-Tensor**: one pair of quantization parameters (`S, Z`) is used per activation tensor
* **Per-Token**: one pair of quantization parameters (`S, Z`) is used per token of the activation tensor. For LLMs, the activation tensor is of shape `[batch_size, seq_len, hidden_dim]`, so the scales will be a matrix of shape `[batch_size, seq_len]`.

Incresing quantization granularity typically helps with accuracy at the expense of less memory reduction and slower inference performance. 

In general, it is best practice to start your experiments with:
- For static activation quantization, always use `per-tensor`
- For `fp8` dynamic quantization, use `per-tensor`
- For `int8` dynamic quantization, use `per-token`

### Activation Reodering

Activations of LLMs are known to be problematic to work with because for some inputs they exhibit very large activation values in a few channels relative to all other channels. Those very large activations are called "outliers", and preserving their propagation through the model is of high importance for good accuracy. Activation reordering triggers quantizing weight columns in order of decreasing activation size, meaning that we first focus on quantizing those that correspond to outliers (to preserve them as good as possible), and then we move on to the others (which correspond to smaller activations by magnitude). This can help preserve accuracy at the expense of some inference speed.

In general it is best practice to start your experients with:
- For `int4` weights, use activation reordering with GPTQ
- For anything else, do not sue activation reordering
