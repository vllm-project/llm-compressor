# Quantizing Mixed-Precision Models with Multiple Quantization Modifiers #

This section outlines how multiple quantization modifiers can be applied to the same model for mixed-precision quantization, for example applying AWQ W4A16 to a model's `self_attn` layers and GPTQ W8A8 to its `mlp` layers. This heterogeneous application of multiple modifiers comes in 2 flavors:

1. Run every modifier in a single, sequential pipeline, performing a single calibrated run. See `./llama3_example.py` for an example.
2. Run each modifier in its own, independent pipeline, performing a calibrated run for each modifier. To run each modifier independently, run `./llama3_example.py` with `oneshot(..., pipeline="independent")` instead of `pipeline="sequential"`.

This is an advanced usage of `llm-compressor` and an active area of research. Best practices will be provided in a future release, after further research and sensitivity analysis.