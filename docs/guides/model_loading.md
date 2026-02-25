# Model Loading #
LLM Compressor utilizes the [Compressed Tensors](https://github.com/vllm-project/compressed-tensors) library to handle model offloading. In nearly all cases, it is recommended to compress your model using the [sequential pipeline](./sequential_onloading.md), which enables the quantization of large models without requiring significant VRAM. 

!!! tip
    For more information on when to use the *basic* pipeline rather than the *sequential* pipeline, see [Basic Pipeline](./model_loading.md#basic-pipeline). In these cases, it is recommended to load your model onto GPU first, rather than CPU/Disk.

Loading your model directly onto CPU is simple using `transformers`:

```python
# model is on cpu
model = AutoModelForCausalLM.from_pretrained(model_stub, dtype="auto")
```

However, there are some exceptions when it is required to change this logic to handle more advanced loading. The table below shows the behavior of different model loading configurations.

Distributed=False | device_map="auto" | device_map="cuda" | device_map="cpu" | device_map="auto_offload"
-- | -- | -- | -- | --
`load_offloaded_model` context required? | No | No | No | Yes
Behavior | Try to load model onto all visible cuda devices. Fallback to cpu and disk if model too large | Try to load model onto first cuda device only. Error if model is too large | Try to load model onto cpu. Error if the model is too large | Try to load model onto cpu. Fallback to disk if model is too large
LLM Compressor Examples | This is the recommended load option when using the "basic" pipeline |   |   | This is the recommended load option when using the "sequential" pipeline

Distributed=True | device_map="auto" | device_map="cuda" | device_map="cpu" | device_map="auto_offload"
-- | -- | -- | -- | --
`load_offloaded_model` context required? | Yes | Yes | Yes | Yes
Behavior | Try to load model onto device 0, then broadcast replicas to other devices. Fallback to cpu and disk if model is too large | Try to load model onto device 0 only, then broadcast replicas to other devices. Error if model is too large | Try to load model onto cpu. Error if the model is too large | Try to load model onto cpu. Fallback to disk if model is too large
LLM Compressor Examples | This is the recommended load option when using the "basic" pipeline |   |   | This is the recommended load option when using the "sequential" pipeline

## Disk Offloading ##
When compressing models which are larger than the available CPU memory, it is recommended to utilize disk offloading for any weights which cannot fit on the cpu. To enable disk offloading, use the `load_offloaded_model` context from `compressed_tensors` to load your model, along with `device_map="auto_offload"`.

```python
from compressed_tensors.offload import load_offloaded_model

with load_offloaded_model():
    model_id = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto_offload",  # fit as much as possible on cpu, rest goes on disk
        max_memory={"cpu": 6e8},  # optional argument to specify how much cpu memory to use
        offload_folder="./offload_folder",  # file system with lots of storage
    )
```

In order to specify where disk-offloaded weights should be stored, please specify the `offload_folder` argument.

You can then call `oneshot` as usual to perform calibration and compression. Some operations may be slower due to disk offloading.

## Distributed Oneshot ##
When performing `oneshot` with distributed computing, you will need to ensure that your model does not replicate offloaded values across ranks, otherwise this will create excess work and memory usage. Coordinated loading between ranks is automatically handled by the `load_offloaded_model` context, so long as it is entered after `torch.distributed` has been initialized.

```python
from compressed_tensors.offload import init_dist, load_offloaded_model

init_dist()
with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", device_map="auto_offload"
    )
```

## Basic Pipeline ##
It is recommended to only use the basic pipeline when your model is small enough to fit into the available VRAM, including any auxillary memory requirements of algorithms such as GPTQ hessians. The basic pipeline can provide compression runtime speedups when compared to the sequential pipeline.

In these cases, you can load the model directly onto your GPU devices, and call oneshot with the relevant argument.

```python
model = AutoModelForCausalLM.from_pretrained(model_stub, device_map="auto")  # model is on devices
...
oneshot(model, ..., pipeline="basic")
```