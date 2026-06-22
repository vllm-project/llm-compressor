# Entrypoints

LLM Compressor provides two entrypoints for post-training quantization (PTQ), each suited to different scenarios. The `compressed-tensors` library also provides a `convert_checkpoint` entrypoint for converting checkpoint formats without loading the model.

## Choosing an Entrypoint

| | [`oneshot`](oneshot.md) | [`model_free_ptq`](model-free-ptq.md) |
|---|---|---|
| **Can apply calibration data** | Yes | No — data-free only |
| **Requires HF model definition** | Yes | No |
| **Supports GPTQ / AWQ / SmoothQuant** | Yes | No |
| **Supports FP8 / NVFP4 data-free** | Yes | Yes |
| **Works when model has no transformers definition** | No | Yes |
| **Fallback when `oneshot` fails** | — | Yes |

## oneshot

Use `oneshot` when your quantization algorithm or scheme **requires calibration data**, such as GPTQ, AWQ, SmoothQuant, or static activation quantization (FP8 or INT8 with static per tensor activations). It loads the model through Hugging Face `transformers`, runs calibration forward passes, and applies recipe-defined modifiers.

[:octicons-arrow-right-24: oneshot documentation](oneshot.md)

## model_free_ptq

Use `model_free_ptq` when your quantization scheme is **data-free** (e.g. FP8 dynamic, FP8 block, NVFP4A16) and either the model has no Hugging Face model definition, or `oneshot` fails for your model. It works directly on the safetensors checkpoint without loading the model through `transformers`.

[:octicons-arrow-right-24: model_free_ptq documentation](model-free-ptq.md)

## convert_checkpoint

Use `convert_checkpoint` (from `compressed-tensors`) when you have a checkpoint in a format such as ModelOpt NVFP4 or AutoAWQ that needs to be converted to compressed-tensors format, or when a compressed-tensors checkpoint needs to be converted to dense weights before re-quantization. It operates entirely on safetensors files without loading the model.

[:octicons-arrow-right-24: convert_checkpoint documentation](convert.md)
