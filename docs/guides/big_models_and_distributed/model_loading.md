# Model Loading #
LLM Compressor utilizes the [Compressed Tensors](https://github.com/vllm-project/compressed-tensors) library to handle model offloading. In nearly all cases, it is recommended to compress your model using the [sequential pipeline](./sequential_onloading.md), which enables the quantization of large models without requiring significant VRAM. 

!!! tip
    For more information on when to use the *basic* pipeline rather than the *sequential* pipeline, see [Basic Pipeline](./model_loading.md#basic-pipeline). In these cases, it is recommended to load your model onto GPU first, rather than CPU/Disk.

Loading your model directly onto CPU is simple using `transformers`:

```python
# model is on cpu
model = AutoModelForCausalLM.from_pretrained(model_stub)
```

However, there are some exceptions when it is required to change this logic to handle more advanced loading. The table below shows the behavior of different model loading configurations.

Distributed=False | device_map="auto" | device_map="cuda" | device_map="cpu" | device_map="auto_offload"
-- | -- | -- | -- | --
`load_context` context required? | No | No | No | Yes
Behavior | Try to load model onto all visible cuda devices. Fallback to cpu and disk if model too large | Try to load model onto first cuda device only. Error if model is too large | Try to load model onto cpu. Error if the model is too large | Try to load model onto cpu. Fallback to disk if model is too large
LLM Compressor Examples | This is the recommended load option when using the "basic" or "data_free" pipeline |   |   | This is the recommended load option when using the "sequential" pipeline

Distributed=True | device_map="auto" | device_map="cuda" | device_map="cpu" | device_map="auto_offload"
-- | -- | -- | -- | --
`load_context` context required? | Yes | Yes | Yes | Yes
Behavior | Try to load model onto device 0, then broadcast replicas to other devices. Fallback to cpu and disk if model is too large | Try to load model onto device 0 only, then broadcast replicas to other devices. Error if model is too large | Try to load model onto cpu. Error if the model is too large | Try to load model onto cpu. Fallback to disk if model is too large
LLM Compressor Examples | This is the recommended load option when using the "basic" or "data_free" pipeline |   |   | This is the recommended load option when using the "sequential" pipeline

## Disk Offloading ##
When compressing models which are larger than the available CPU memory, it is recommended to utilize disk offloading for any weights which cannot fit on the cpu. To enable disk offloading, use the `load_context` context from `llmcompressor.utils` to load your model, along with `device_map="auto_offload"`.

```python
from llmcompressor.utils import load_context

with load_context():
    model_id = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto_offload",  # fit as much as possible on cpu, rest goes on disk
        max_memory={"cpu": 6e8},  # optional argument to specify how much cpu memory to use
        offload_folder="./offload_folder",  # file system with lots of storage
    )
```

In order to specify where disk-offloaded weights should be stored, please specify the `offload_folder` argument.

You can then call `oneshot` as usual to perform calibration and compression. Some operations may be slower due to disk offloading.

## Distributed Oneshot ##
When performing `oneshot` with distributed computing, you will need to ensure that your model does not replicate offloaded values across ranks, otherwise this will create excess work and memory usage. Coordinated loading between ranks is automatically handled by the `load_context` context, so long as it is entered after `torch.distributed` has been initialized.

```python
from compressed_tensors.offload import init_dist
from llmcompressor.utils import load_context

init_dist()
with load_context():
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto_offload")
```

## Multi-Token Prediction (MTP) ##

A checkpoint can contain MTP weights without those layers being loaded into the model. `has_mtp_in_checkpoint(model)` checks whether the config declares MTP layers. `hasattr(model, "mtp")` checks whether the model has an attached MTP module.

The [MTP examples](https://github.com/vllm-project/llm-compressor/tree/main/examples/mtp) demonstrate the following workflows:

- [GLM-4.7-Flash](https://github.com/vllm-project/llm-compressor/blob/main/examples/mtp/glm4_7_flash_copy_mtp.py): quantize the backbone to FP8 with `load_context(load_mtp=False)` and copy the source BF16 MTP tensors unchanged when saving.
- [GLM-4.5-Air](https://github.com/vllm-project/llm-compressor/blob/main/examples/mtp/glm4_5_air_mtp.py): load MTP with `load_context(load_mtp=True)` and apply the same `FP8_DYNAMIC` scheme to the backbone and MTP.
- [DeepSeek V3](https://github.com/vllm-project/llm-compressor/blob/main/examples/mtp/deepseek_v3_nvfp4_fp8_mtp.py): use separate `QuantizationModifier` instances and the independent pipeline to calibrate the backbone to NVFP4, then quantize the MTP decoder to `FP8_DYNAMIC` without calibration data. Treat `MtpModel` as a tracing boundary and update the exported scheme targets to match checkpoint names without overlap.
- [GLM-5.3](https://github.com/vllm-project/llm-compressor/blob/main/examples/model_free_ptq/glm_5_3_mxfp4.py): use `model_free_ptq` to convert source FP8 checkpoint weights and quantize the backbone and MTP to MXFP4. `FP8BlockDequantizer.from_pretrained()` identifies source FP8 tensors from their scale entries.

### Loading and quantizing MTP

1. `load_context()` defaults to `load_mtp=True`. During `from_pretrained()`, `load_with_mtp_model()` loads supported MTP layers with Transformers' `MtpModel` and attaches them as `model.mtp` before MoE conversion and offloading. Models whose config does not declare MTP load normally.
2. `_finalize_mtp_weight_conversions()` preserves the mapping between MTP module names and checkpoint names after MoE conversion, so saving can restore the original checkpoint names.
3. Before applying the recipe, `oneshot()` calls `validate_model()` to check its targets and ignores. Quantization targeting MTP must use a data-free scheme such as `FP8_DYNAMIC`; methods requiring calibration are rejected. The recipe then applies to matching backbone and MTP modules.
4. `save_pretrained()` serializes loaded MTP with the rest of the model.

Recipe targets select the loaded Hugging Face modules. Saving restores checkpoint names for weights and ignore entries, but leaves quantization targets unchanged. For example, a target such as `re:^mtp\.layers\.` can select MTP during quantization while the saved weights use `model.layers.N.*` or `model.mtp.layers.*`.

If named targets no longer match after saving, update the targets in `config.json` to cover the checkpoint names and any prefix changes made by vLLM while preserving each module's scheme. Ensure backbone and MTP targets do not overlap when their schemes differ. The shared `FP8_DYNAMIC` scheme with `Linear` targets in the GLM-4.5-Air example does not require this correction.

### Preserving unloaded MTP

Use `load_context(load_mtp=False)` to skip attaching MTP. With `load_mtp=True`, missing checkpoint patterns or a config rejected by Transformers' `get_mtp_config()` also leave MTP unloaded, with a warning. Errors during actual MTP weight loading still propagate. `oneshot()` does not load these layers later. Its `validate_model()` calls `validate_mtp_copy_source()` to check the source weights and warns that MTP will be copied unchanged. The recipe only affects loaded modules.

The save wrapper also calls `validate_mtp_copy_source()` before writing any checkpoint files. This prevents a validation failure from leaving a partial checkpoint, including when calling `save_pretrained()` directly. After saving the backbone, it passes the validated source to `save_mtp_tensors()`, which copies the MTP weights unchanged and adds their module names to the exported quantization ignore list. If the config declares MTP but no weights are found, saving warns that there is nothing to copy.

Loading through `MtpModel` and copying unloaded MTP both reject MTP weights that are already quantized in the source checkpoint. To convert these checkpoints or quantize unsupported MTP layouts, use an explicit model-free workflow; see the [model-free conversion example](https://github.com/vllm-project/llm-compressor/blob/main/examples/model_free_ptq/glm_5_3_mxfp4.py), which works directly on checkpoint tensors. This fallback never runs model-free quantization automatically.

## Basic Pipeline ##
The basic pipeline calibrates the whole model without tracing. Use it only when the model and calibration memory, such as GPTQ hessians, fit in the available VRAM.

In these cases, you can load the model directly onto your GPU devices, and call oneshot with the relevant argument.

```python
model = AutoModelForCausalLM.from_pretrained(model_stub, device_map="auto")  # model is on devices
...
oneshot(model, ..., pipeline="basic")
```
