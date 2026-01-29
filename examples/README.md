# LLM Compressor Examples

The LLM Compressor examples are organized primarily by quantization scheme, which is indicated by the folder suffix `quantization_***`. Each of these folders contains model-specific examples showing how to apply that quantization scheme to a particular model.

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

If you’re unsure which quantization scheme to use, a good starting point is a data-free pathway, such as `w8a8_fp8`, found under `quantization_w8a8_fp8`. For more details on available schemes and when to use them, see the
[Compression Schemes guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/compression_schemes/).

## Need help?

If you don’t see your model or aren’t sure which quantization scheme applies, feel free to open an issue and someone from the community will be happy to help.

Note: We are currently in the process of updating and improving our documentation and examples structure. Feedback is very welcome during this transition.

# Quantizing MoEs

Quantizing MoE models with a scheme that requires calibration data (for example, schemes where activations are not dynamic, such as FP8 or INT8 per-tensor activations, or NVFP4), or with an algorithm that requires data (such as GPTQ, AWQ, or AutoRound), requires a calibration-friendly MoE block definition for the model being quantized.

Examples of these calibration-friendly definitions can be found in the [modeling folder](../src/llmcompressor/modeling/). Each definition enables an MoE calibration context by inheriting from the [`MoECalibrationModule` class](../src/llmcompressor/modeling/moe_context.py) and registering the MoE block that should be replaced with a custom definition.

In particular, each model-specific definition includes an updated forward pass that ensures all tokens are routed through all experts during calibration, including experts that would not normally be activated. Only the activated experts, however, contribute to the final output of the MoE block. This behavior ensures proper calibration of all expert layers.

These custom definitions replace the existing MoE implementations during `oneshot`. The replacement can be either temporary or permanent; in the temporary case, the original definition is restored after calibration. One example is the `qwen3_vl_moe` custom MoE definition, which registers to replace all `Qwen3MoeSparseMoeBlock` instances with `CalibrationQwen3MoeSparseMoeBlock`. You can see this example [here](../src/llmcompressor/modeling/qwen3_vl_moe.py).

Without a custom calibration-friendly definition, MoE experts may be calibrated incorrectly, which can result in numerical instability or NaNs.
