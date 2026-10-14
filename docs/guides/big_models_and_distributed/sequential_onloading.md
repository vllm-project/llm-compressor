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

## Native disk staging (experimental) ##

The sequential pipeline prefetches the next disjoint group of modules on one
background thread. An optional native backend releases the Python GIL across
the batch of file opens, safetensors header parsing, CPU tensor allocations,
disk reads, and dtype conversions. The main thread can continue submitting GPU
work during that batch. Python still prepares requests and installs tensors in
the module caches before the staging future completes.

Build against the PyTorch already installed in your environment (Linux, C++17
compiler and Python development headers required):

```bash
python -m pip install setuptools wheel setuptools_scm==8.2.0
LLMCOMPRESSOR_BUILD_NATIVE_STAGING=1 python -m pip install --no-build-isolation --no-deps -e .
```

Rebuild after changing PyTorch or Python. Normal installs remain pure Python.
When the extension is available, disk staging uses it automatically and logs
`Using native disk staging with GIL release`. Set
`LLMCOMPRESSOR_NATIVE_STAGING=0` to compare against Python staging without
rebuilding. There is no runtime compilation.

This works with both values of `stage_weights_in_pinned_memory`. Reads go directly
into individually allocated CPU tensors, pinned when requested for accelerator
execution. There is no reusable buffer pool. Dtype conversions use a temporary
CPU tensor when the stored dtype differs from the requested dtype. The existing
onload path consumes these tensors on its current accelerator stream; PyTorch
owns their allocation and asynchronous transfer lifetime.

CPU caches, custom/distributed caches, tensor subclasses, unsupported source
dtypes, and installations without the extension use the existing staging path.
The extension requires little-endian hosts. Disk errors propagate through the
staging future; native results are installed only after the batch succeeds.
Native staging can still contend with the foreground for allocator locks,
memory bandwidth, and disk bandwidth. Measure the complete calibration run as
well as staging time.

## More information ##

For more information, see the [RedHat AI blog post](https://developers.redhat.com/articles/2025/05/09/llm-compressor-optimize-llms-low-latency-deployments#generalizing_to_multimodal_and_moe_architectures) or the [LLM Compressor Office Hours Recording](https://www.youtube.com/watch?v=GrhuqQDmBk8).
