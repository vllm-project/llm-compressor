# Quantizing models without a model definition 

`model_free_ptq` provides a PTQ pathway for data-free schemes (such for FP8 Dynamic Per Token or FP8 Block). Specifically, this pathway removes the requirement for a model definition or the need to load the model through transformers. If you are interested in applying a data-free scheme, there are two key scenarios in which applying this pathway may make sense for your model:

1. The model does not have a model definition available through transformers. This may be the case for a brand new model which has not landed in transformers.
2. The model is very large (such as Kimi K2 Thinking) and is running into issues with `oneshot`


`model_free_ptq` works directly with the safetensors in the checkpoint to which observers are applied, thereby removing the requirement for a model definition or transformers.

# Quantizing Kimi K2 Thinking to FP8 Block 

In `kimi_k2_thinking_fp8_block.py`, we call `model_free_ptq` by providing a `scheme` and `ignore` list, similar to how we provide reicpes to `oneshot` calls. In the case of Kimi-K2 Thinking, we apply the `FP8_BLOCK` scheme and ignore layers that are incompatible with a block_size of 128 (specifically, `kv_a_proj_with_mqa` and `q_a_proj`).

In contrast to `oneshot`, we expect the model stub or pathway string to be directly passed in, as opposed to first being loaded through transformers. Once complete, the model is compressed using compressed-tensors and saved to `SAVE_DIR`.
