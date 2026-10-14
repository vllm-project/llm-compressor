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

## Staging in a Python process ##

The sequential pipeline prefetches the next disjoint subgraph using a persistent
Python worker process. The worker parses safetensors headers and reads tensor
payloads while the parent continues calibration. Its interpreter has a separate
GIL, so Python execution in the worker can run in parallel with the parent's
Python execution. The staging thread in the parent submits batches and receives
their results.

Process staging is enabled automatically on little-endian Linux hosts with
`os.preadv` and `os.posix_fallocate`. Other platforms, CPU caches,
custom/distributed caches and tensor subclasses retain the upstream staging
behavior. Unsupported source dtypes also use the upstream reader.

Each batch is written to a fresh anonymous shared-memory file. The worker sends
one file descriptor and tensor metadata to the parent, without serializing the
tensor payload through the pipe. It receives only filenames and tensor keys;
the model and CUDA stay in the parent. The worker imports only the standard
library and runs as a standalone script, so callers do not need a multiprocessing
`if __name__ == "__main__"` guard. There is no extra compilation step.

The parent wraps shared memory as CPU tensors, restores module metadata, and
copies into pinned memory when `stage_weights_in_pinned_memory=True` for an
accelerator. These operations still run in the parent interpreter. With pinning
disabled and no dtype conversion, tensors keep the shared mapping alive without
another payload copy. Pinning or dtype conversion needs an additional allocation
and copy, so separate GILs do not guarantee faster total calibration time.

By default, shared files use `/dev/shm`. It must hold a staging batch (about 7 GB
for four decoder layers in the Gemma-4-26B-A4B example); pinned allocations can
coexist with those shared pages during copying. Use
`LLMCOMPRESSOR_STAGING_SHM_DIR` to select another suitable directory if necessary.
A directory backed by disk can add I/O costs. Space is reserved before writing
so allocation failures propagate as errors. There is no retained buffer pool.

The worker persists across subgraphs. On exit, including errors, the pipeline
joins the staging thread before closing the worker. Shared files are anonymous
and close with their last mapping/descriptor; pageable tensor views can outlive
the worker. Read failures propagate through the staging future before the new
batch is installed in the module caches.

## More information ##

For more information, see the [RedHat AI blog post](https://developers.redhat.com/articles/2025/05/09/llm-compressor-optimize-llms-low-latency-deployments#generalizing_to_multimodal_and_moe_architectures) or the [LLM Compressor Office Hours Recording](https://www.youtube.com/watch?v=GrhuqQDmBk8).
