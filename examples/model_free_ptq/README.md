# Model-free Quantization

`model_free_ptq` provides a PTQ pathway for data-free schemes (such for FP8 Dynamic Per Token or FP8 Block). Specifically, this pathway removes the requirement for a model definition or the need to load the model through transformers. If you are interested in applying a data-free scheme, there are two key scenarios in which applying this pathway may make sense for your model:

1. The model does not have a model definition available through transformers. This may be the case for a brand new model which has not landed in transformers.
2. The model is very large (such as Kimi K2 Thinking) and is running into issues with `oneshot`


`model_free_ptq` works directly with the safetensors in the checkpoint to which observers are applied, thereby removing the requirement for a model definition or transformers.

# Quantizing Kimi K2 Thinking to FP8 Block 

In `kimi_k2_thinking_fp8_block.py`, we call `model_free_ptq` by providing a `scheme` and `ignore` list, similar to how we provide reicpes to `oneshot` calls. In the case of Kimi-K2 Thinking, we apply the `FP8_BLOCK` scheme and ignore layers that are incompatible with a block_size of 128 (specifically, `kv_a_proj_with_mqa` and `q_a_proj`).

In contrast to `oneshot`, we expect the model stub or pathway string to be directly passed in, as opposed to first being loaded through transformers. Once complete, the model is compressed using compressed-tensors and saved to `SAVE_DIR`.

To get started, simply call `model_free_ptq` with your desired model stub and save directory
```python
model_free_ptq(
    model_stub="unsloth/Kimi-K2-Thinking-BF16",
    save_directory="Kimi-K2-Thinking-FP8-BLOCK",
    scheme="FP8_BLOCK",
    ignore=[
        "re:.*gate$",
        "lm_head",
        "re:.*kv_a_proj_with_mqa$",
        "re:.*q_a_proj$",
        "model.embed_tokens",
    ],
    max_workers=15,
    device="cuda:0",
)

```


# Quantizing models to NVFP4A16/ MXFP4A16

Using model_free_ptq to quantize models with microscale schemes (NVFP4/MXFP4) is the same as quantizing models with non-microscale schemes, except for one additional step. That extra step is that the safetensors in the model files must be reindexed to ensure that fused modules (qkv, gate_up) end up in the same safetensors files, which allows model_free_ptq to fuse global scales.

First, apply `llmcompressor.reindex_fused_weights` from the command line entrypoint
```bash
llmcompressor.reindex_fused_weights \
    unsloth/Kimi-K2-Thinking-BF16 \
    Kimi-K2-Thinking-BF16-reindexed \
    --num_workers=10
```

Then, call `model_free_ptq` on the reindex files
```python
model_free_ptq(
    model_stub="Kimi-K2-Thinking-BF16-reindexed",
    save_directory="Kimi-K2-Thinking-BF16-NVFP4A16",
    scheme="NVFP4A16",
    ignore=[
        "re:.*gate$",
        "lm_head",
        "re:.*kv_a_proj_with_mqa$",
        "re:.*q_a_proj$",
        "model.embed_tokens",
    ],
    max_workers=15,
    device="cuda:0",
)
```