---
weight: -4
---

# LLM Compressor Examples

The LLM Compressor examples are organized primarily by quantization scheme. Each folder contains model-specific examples showing how to apply that quantization scheme to a particular model.

Some examples are additionally grouped by model type, such as:
- `multimodal_audio`
- `multimodal_vision`
- `quantizing_moe`

Other examples are grouped by algorithm, such as:
- `awq`
- `autoround`

## How to find the right example

- If you are interested in quantizing a specific model, start by browsing the model-type folders (for example, `multimodal_audio`, `multimodal_vision`, or `quantizing_moe`).
- If you don’t see your model there, decide which quantization scheme you want to use (e.g., FP8, FP4, INT4, INT8, or KV cache / attention quantization) and look in the corresponding `quantization_***` folder.
- Each quantization scheme folder contains at least one LLaMA 3 example, which can be used as a general reference for other models.

## Where to start if you’re unsure

If you’re unsure which quantization scheme to use, a good starting point is a data-free pathway, such as `w8a8_fp8`, found under `quantization_w8a8_fp8`. For more details on available schemes and when to use them, see the Compression Schemes [guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/compression_schemes/).

## Need help?

If you don’t see your model or aren’t sure which quantization scheme applies, feel free to open an issue and someone from the community will be happy to help.

!!! note
    We are currently updating and improving our documentation and examples structure. Feedback is very welcome during this transition.