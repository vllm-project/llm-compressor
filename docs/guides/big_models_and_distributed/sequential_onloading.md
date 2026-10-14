# Sequential Onloading #

## Introduction ##

LLM Compressor is capable of compressing models much larger than the amount of memory available as VRAM. This is achieved through a technique called **sequential onloading** whereby only a fraction of the model weights are moved to GPU memory for calibration while the rest of the weights remain offloaded to CPU or disk. When performing calibration, the entire dataset is offloaded to CPU, then onloaded one batch at a time to reduce peak activations memory usage.

![sequential_onloading](../../assets/sequential_onloading.jpg)

If basic calibration/inference is represented with the following pseudo code...
```python
for i in range(len(activations)):
    for layer in model.layers:
        activations[i] = layer(activations[i])
```

Then sequential onloading is the technique by which the order of the two for loops is swapped.
```python
for layer in model.layers:
    for i in range(len(activations)):
        activations[i] = layer(activations[i])
```

## Implementation ##

Before a model can be sequentially onloaded, it must first be broken up into disjoint parts which can be individually onloaded. This is achieved through the [torch.fx.Tracer](https://github.com/pytorch/pytorch/blob/main/torch/fx/README.md#tracing) module, which allows a model to be represented as a graph of operations (nodes) and data inputs (edges). Once the model has been traced into a valid graph representation, the graph is cut (partitioned) into disjoint subgraphs, each of which is onloaded individually as a layer. This implementation can be found [here](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/pipelines/sequential/helpers.py).

![sequential_onloading](../../assets/model_graph.jpg)
*This image depicts some of the operations performed when executing the Llama3.2-Vision model*

![sequential_onloading](../../assets/sequential_decoder_layers.jpg)
*This image depicts the sequential text decoder layers of the Llama3.2-Vision model. Each of the individual decoder layers is onloaded separately*

## Sequential Targets and Usage ##
You can use sequential onloading by calling `oneshot` with the `pipeline="sequential"` argument. Note that this pipeline is the default for all oneshot calls which require calibration data. If the sequential pipeline proves to be problematic, you can specify `pipeline="basic"` to use a basic pipeline which does not require sequential onloading, but only works performantly when the model is small enough to fit into the available VRAM.

If you are compressing a model using a GPU with a small amount of memory, you may need to change your sequential targets. Sequential targets control how many weights to onload to the GPU at a time. By default, the sequential targets are decoder layers which may include large MoE layers. In these cases, setting the `sequential_targets="Linear"` argument in `oneshot` will result in lower VRAM usage, but a longer runtime.

![sequential_onloading](../../assets/seq_targets.jpg)

## Disk staging ##

The sequential pipeline prefetches the next disjoint subgraph on a background
thread. For standard disk caches on little-endian hosts with `os.preadv`, it
reads safetensors payloads directly into PyTorch CPU allocations. Tensors from
the same resolved shard share header parsing within each staging batch.

Direct reads are enabled automatically and work with either value of
`stage_weights_in_pinned_memory`. When pinning is enabled for accelerator
execution, reads go directly into pinned storage. Dtype conversion uses a
temporary source tensor when the stored dtype differs from the requested dtype.
Each tensor has its own allocation, managed by PyTorch through the subsequent
onload; staging does not retain a separate buffer pool.

CPU caches, custom/distributed caches, tensor subclasses, unsupported source
dtypes and unsupported platforms use the existing compressed-tensors reader.
Read errors propagate through the staging future, and direct-read cache mappings
are published after the batch succeeds.

This path uses Python and existing PyTorch/NumPy APIs with no additional build
step. Python metadata handling and the per-tensor loop still share the GIL with
calibration, so it does not eliminate all thread contention.

## More information ##

For more information, see the [RedHat AI blog post](https://developers.redhat.com/articles/2025/05/09/llm-compressor-optimize-llms-low-latency-deployments#generalizing_to_multimodal_and_moe_architectures) or the [LLM Compressor Office Hours Recording](https://www.youtube.com/watch?v=GrhuqQDmBk8).
