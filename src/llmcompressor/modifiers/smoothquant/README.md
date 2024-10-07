# SmoothQuant Modifier Mapping Tutorial

In this tutorial, we'll cover how to specify the correct mappings for applying the SmoothQuant Modifier from the [LLM Compressor](https://github.com/vllm-project/llm-compressor) repository, based on the SmoothQuant paper [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438).

## Understanding the Mapping Format

### Context
SmoothQuant leverages activation scaling to smooth out input activations, making quantization more efficient for large language models (LLMs). As mentioned in the SmoothQuant paper, "By default, we perform scale smoothing for the input activations of self-attention and feed-forward layers."

This means that we need to smooth the inputs feeding into:
- The **q/k/v blocks** (query, key, value blocks of self-attention)
- The **fc1 block** (the fully connected block of the feed-forward layer)

We can derive this by examining the diagram on page 5 of the SmoothQuant paper. It shows that smoothing should occur at specific points in the neural network architecture.

### Layer Selection

To get the correct input for smoothing:
1. For **q/k/v blocks**, based on the SmoothQuant paper, we need to target the outputs of `input_layernorm`, as these provide the inputs for the self-attention mechanism.
2. For the **fc1 block**, based on the SmoothQuant paper, we need to target the outputs of `post_attention_layernorm`.

### Why Target Leaf Modules?

Based on the SmoothQuant paper smoothing needs to be applied at the leaf nodes of the computational graph. This is why we see mappings such as:

```python
[["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"]
```

Instead of targeting broader modules like `mlp`, we explicitly specify the lower-level projections (`gate_proj` and `up_proj`) and the `post_attention_layernorm` normalization.

### The Mapping Format

A mapping in SmoothQuant takes the form:

```python
[[layers smoothed input activations pass into], output_to_smooth]
```

For example, in the default mapping:
```python
[["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"]
```
This specifies that we want to smooth the inputs feeding into the projections (`gate_proj`, `up_proj`) and the output from `post_attention_layernorm`.

## Specifying Your Own Mappings

To create your own mappings, follow these steps:

1. **Identify the layers you want to pass smoothed input activations into**:
 You can find the exact names of these layers by exploring the relevant model file (e.g., `modeling_llama.py`). For example, you might target layers related to the self-attention or feed-forward blocks.

2. **Match leaf modules**:
 Ensure you're targeting leaf modules (i.e., the individual components of broader blocks, such as `gate_proj` and `up_proj` instead of a larger `mlp` module).

3. **Specify the correct regular expressions**:
 Use regular expressions to match the layers you want to target. For instance, if you want to target all projection layers across all attention heads, you could use a regex like `"re:.*proj"`. If you want to target a specific projection layer, make the regex more specific.

### Example Custom Mapping

Let's say you're working with a model with layers named similar to LLaMA, and you want to smooth the input activations of the self-attention layers and the feed-forward layers. Here is how you might specify the mapping:

```python
mapping = [
    # Smooth the inputs going into the query, key, value projections of self-attention
 [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
    # Smooth the inputs going into the first feed-forward block (fc1)
 [["re:.*fc1"], "re:.*post_attention_layernorm"]
]
```

In this mapping:
- We are targeting the `q_proj`, `k_proj`, and `v_proj` layers for smoothing by using the outputs of `input_layernorm`.
- We are targeting the `fc1` feed-forward block by using the outputs of `post_attention_layernorm`.

This ensures that SmoothQuant modifies the correct activations, improving quantization efficiency while maintaining model accuracy.

## Conclusion

By understanding the structure of your model and specifying precise mappings, you can apply the SmoothQuant Modifier effectively. Use the diagram on page 5 of the [SmoothQuant paper](https://arxiv.org/pdf/2211.10438) and inspect your model's code to identify the correct layers and leaf modules to target for smoothing.

Now that you know how to create these mappings, you can experiment with different model architectures and observe how SmoothQuant impacts performance and quantization accuracy.