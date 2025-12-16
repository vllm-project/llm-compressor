# Attention Quantization in LLM Compressor #
LLM Compressor supports applying static attention quantization to models. Please note that attention quantization support in vLLM is still ongoing and is not fully supported as of this writing.

## FP8 Attention Example ##
For an example applying attention quantization, see [llama3_attention.py](/experimental/attention/llama3_attention.py).

```python
recipe = QuantizationModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=["LlamaAttention"],
            input_activations=QuantizationArgs(
                num_bits=8, type="float", strategy="attn_head"
            ),
        )
    }
)
```

Note that attention quantization also implicitly applies kv cache quantization with the same quantization arguments.

## NVFP4 Attention + R3 Example ##
Attention quantization can be improved using the R3 transform, as described by [SpinQuant](https://arxiv.org/abs/2405.16406). This transform reduces the presence of outliers in the attention activation distribution, thereby improving accurcy recovery.

```python
recipe = [
    SpinQuantModifier(rotations=["R3"]),
    QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=["LlamaAttention"],
                input_activations=NVFP4["input_activations"],
            )
        }
    ),
]
```

### Evaluations ###
Utilizing the R3 transform has been shown to improve accuracy recovery for the `meta-llama/Llama-3.2-1B-Instruct` model when using NVFP4 attention quantization.

Without R3 Transform
```
../llm-compressor/Llama-3.2-1B-Instruct-attention-nvfp4/
|    Tasks     |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|--------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k_platinum|      3|flexible-extract|     5|exact_match|↑  |0.2680|±  |0.0127|
|              |       |strict-match    |     5|exact_match|↑  |0.1836|±  |0.0111|
```

With R3 Transform
```
../llm-compressor/Llama-3.2-1B-Instruct-r3-attention-nvfp4/
|    Tasks     |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|--------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k_platinum|      3|flexible-extract|     5|exact_match|↑  |0.2961|±  |0.0131|
|              |       |strict-match    |     5|exact_match|↑  |0.2283|±  |0.0121|
```