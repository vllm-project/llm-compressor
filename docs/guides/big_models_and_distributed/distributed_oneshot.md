## Distributed Oneshot ##
LLM Compressor supports distributed oneshot to greatly speed up model calibration and compression. Each rank processes a disjoint partition of the calibration dataset (data-parallel calibration); modifiers then synchronize statistics across ranks so all ranks produce identical quantization parameters. For the GPTQ modifier, compression itself is also distributed: modules are assigned to ranks by a greedy bin-packing algorithm and the compressed weights are broadcast back. For more information on the design, see [[RFC] [Performance Refactor][Distributed] Sequential Onloading with Data-Parallel Calibration and Weight-Parallel Optimization](https://github.com/vllm-project/llm-compressor/issues/2180) as well as [[GPTQ][ddp] enabling DDP for GPTQ](https://github.com/vllm-project/llm-compressor/pull/2333).

## Usage ##
In order to convert a script meant for single-threaded compression into one of distributed compression, please make the following changes:

### 1. Initialize the Distributed Context ###

In order to utilize the `torch.distributed` module, each rank must initialize the distributed module and assign itself a separate GPU device. This can be done by calling the `init_dist` utility provided by `compressed_tensors`. 

```python
from compressed_tensors.offload import init_dist

init_dist()
```

### 2. Modify Model Loading ###

In order to prevent separate processes from loading the model multiple times and creating excess work/memory usage, we must load our model using the `load_offloaded_model` context. For more information, see [Model Loading](./model_loading.md#distributed-oneshot).

Before:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto"
)
```

After:
```python
from compressed_tensors.offload import load_offloaded_model

with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto_offload",
    )
```

### 3. Modify Dataset Loading ###

In order to prevent separate processes loading the entire dataset and creating excess work/memory usage, we must partition our dataset into disjoint sets. For a dataset of *N* samples and *R* ranks, each rank only loads *N/R* samples.

```python
ds = load_dataset(
    DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"
)
```


```python
from llmcompressor.datasets.utils import get_rank_partition

ds = load_dataset(
    DATASET_ID, split=get_rank_partition(DATASET_SPLIT, NUM_CALIBRATION_SAMPLES)
)
```

Alternatively, if you pass a `Dataset` object directly to `oneshot`, the sampler automatically detects when all ranks have the same dataset (via fingerprint comparison) and partitions it across ranks for you — no manual partitioning needed in that case.

### 4. Call your script with `torchrun` ###

Now, your script is ready to run using distributed processes. To start, simply run your script using `torchrun --nproc_per_node=2 YOUR_EXAMPLE.py` to run with two GPU devices.

### 5. Post-quantization: dispatch and save ###

After `oneshot` returns, call `dispatch_model` to move the model to a single GPU for inference or validation:

```python
from compressed_tensors.offload import dispatch_model

dispatch_model(model)
```

To save the model, `save_pretrained` handles distributed internals automatically — only rank 0 writes to disk. You should still call it from all ranks (or guard on rank 0 for the save path), and call `destroy_process_group()` at the end:

```python
import torch.distributed as dist

model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

dist.destroy_process_group()
```

## Quantization Modifier DDP Support ##

The following quantization modifiers are DDP-aware. Each section describes its synchronization strategy.

### GPTQModifier ###

GPTQ uses **weight-parallel compression**: after each rank accumulates Hessians from its data partition, the modifier:

1. Uses a greedy bin-packing algorithm to assign modules to ranks (balanced by Hessian size).
2. Each rank reduces (`dist.reduce`) the Hessians for modules assigned to it, accumulating contributions from all ranks.
3. Each rank compresses only its assigned modules independently.
4. Compressed parameters (`weight`, `weight_scale`, `weight_zero_point`) are broadcast back to all ranks via `dist.broadcast`.

This means compression itself is parallelized across ranks, not just calibration.

See [llama3_ddp_example.py](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a16/llama3_ddp_example.py) for a complete W4A16 example.

**Benchmark results** (as of LLM Compressor v0.10.0):

| model_id | world_size | max_time | max_memory | save_time | flex_extract | eval_time |
|----------|-------------|----------|------------|-----------|--------------|-----------|
| Meta-Llama-3-8B-Instruct |  1 | 745.03 | 5.82 | 19.57 | 0.7066 | 95.28 |
| Meta-Llama-3-8B-Instruct | 2 | 372.20 | 5.57 | 49.10 | 0.7089 | 95.24 |
| Meta-Llama-3-8B-Instruct | 4 | 264.07 | 5.82 | 52.50 | 0.7180 | 96.74 |
| Qwen3-30B-A3B | 1 | 14207.53 | 6.56 | 748.23 | 0.8704 | 209.93 |
| Qwen3-30B-A3B | 2 | 7018.25 | 6.36 | 696.65 | 0.8810 | 205.89 |
| Qwen3-30B-A3B | 4 | 3694.46 | 6.36 | 723.05 | 0.8832 | 217.62 |

### QuantizationModifier (activation quantization / W8A8) ###

When using activation quantization (e.g., W8A8 with `QuantizationModifier`), activation observer statistics are **all-reduced across ranks at sequential layer boundaries** — so all ranks share identical quantization parameters before moving to the next layer. The reduce operation varies by observer type (see [Observer DDP support](#observer-ddp-support) below).

### AutoRoundModifier ###

AutoRound supports DDP by partitioning the calibration dataset across ranks. The same model-load and dataset-partition pattern applies. See [ddp_qwen3_example.py](https://github.com/vllm-project/llm-compressor/blob/main/examples/autoround/ddp/ddp_qwen3_example.py) for a complete example.

## Transform Modifier DDP Support ##

Transform modifiers (which rescale weights and activations before quantization) are also DDP-aware. Each section describes its synchronization strategy.

### SmoothQuantModifier ###

SmoothQuant uses **all-reduce of per-channel activation statistics** across ranks. Each rank observes its partition of the calibration data to collect per-channel min/max values. Before smoothing scales are computed, `dist.all_reduce` (with MIN/MAX ops) ensures every rank has the global per-channel statistics. The smoothing scale computation is then performed independently and identically on each rank — no broadcast of the final weight tensors is needed.

See [smoothquant_ddp_example.py](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_int8/smoothquant_ddp_example.py) for a complete W8A8 + SmoothQuant example.

### AWQModifier ###

AWQ uses **all-reduce of activation statistics** across ranks before computing smoothing scales and clipping parameters. Similar to SmoothQuant, each rank observes its data partition and the statistics are globally reduced. The scale computation is then replicated identically on each rank.

See [llama_example_ddp.py](https://github.com/vllm-project/llm-compressor/blob/main/examples/awq/llama_example_ddp.py) for a LLaMA W4A16 example, and [qwen3_moe_example_ddp.py](https://github.com/vllm-project/llm-compressor/blob/main/examples/awq/qwen3_moe_example_ddp.py) for a Qwen3-MoE example.

## Observer DDP Support ##

Observers (used during activation calibration) expose a `sync_activation_stats()` method that all-reduces their accumulated statistics across DDP ranks. Each observer class declares which statistics to sync and with what reduce operation via the `_act_sync_dict` class attribute:

| Observer | Synced statistics | Reduce op |
|----------|-------------------|-----------|
| `static_minmax` | `min_vals`, `max_vals` | MIN, MAX |
| `minmax` (EMA) | `min_vals`, `max_vals` | AVG |
| `memoryless_minmax` | *(none — data-free)* | — |
| `mse` (EMA) | `min_vals`, `max_vals` | AVG |
| `memoryless_mse` | *(none — data-free)* | — |
| `imatrix_mse` | `_imatrix_sum`, `_imatrix_count` | SUM |

Weight statistics are never synced: weights are identical across ranks (broadcast at load time), so only activation-derived statistics need synchronization.

The `QuantizationMixin.sync_obs_act_stats()` method (called at each sequential layer boundary) iterates all observers on the modules in the current subgraph and calls `observer.sync_activation_stats()`, collecting async communication handles and waiting for completion before proceeding to the next layer.

## Complete Example Scripts ##

| Algorithm | Example |
|-----------|---------|
| GPTQ W4A16 | [llama3_ddp_example.py](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a16/llama3_ddp_example.py) |
| SmoothQuant + GPTQ W8A8 | [smoothquant_ddp_example.py](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_int8/smoothquant_ddp_example.py) |
| AWQ + QuantizationModifier W4A16 (LLaMA) | [llama_example_ddp.py](https://github.com/vllm-project/llm-compressor/blob/main/examples/awq/llama_example_ddp.py) |
| AWQ + QuantizationModifier W4A16 (Qwen3 MoE) | [qwen3_moe_example_ddp.py](https://github.com/vllm-project/llm-compressor/blob/main/examples/awq/qwen3_moe_example_ddp.py) |
| AutoRound W4A16 | [ddp_qwen3_example.py](https://github.com/vllm-project/llm-compressor/blob/main/examples/autoround/ddp/ddp_qwen3_example.py) |
