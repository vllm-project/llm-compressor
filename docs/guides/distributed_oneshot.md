## Distributed Oneshot ##
As an experimental feature, LLM Compressor supports distributed oneshot for the purpose of greatly speeding up the runtime of model calibration and compression. For more information on implementation, see [[RFC] [Performance Refactor][Distributed] Sequential Onloading with Data-Parallel Calibration and Weight-Parallel Optimization](https://github.com/vllm-project/llm-compressor/issues/2180) as well as [[GPTQ][ddp] enabling DDP for GPTQ](https://github.com/vllm-project/llm-compressor/pull/2333).

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

### 4. Call your script with `torchrun` ###

Now, your script is ready to run using distributed processes. To start, simply run your script using `python3 -m torchrun --nproc_per_node=2 YOUR_EXAMPLE.py` to run with two GPU devices. For a complete example script, see [llama_ddp_example.py](/examples/quantization_w4a16/llama3_ddp_example.py). The below table shows results and speedups as of LLM Compressor v0.10.0, future changes will bring these numbers closer to linear speedups.

| model_id | world_size | max_time | max_memory | save_time | flex_extract | eval_time |
|----------|-------------|----------|------------|-----------|--------------|-----------|
| Meta-Llama-3-8B-Instruct |  1 | 745.03 | 5.82 | 19.57 | 0.7066 | 95.28 |
| Meta-Llama-3-8B-Instruct | 2 | 372.20 | 5.57 | 49.10 | 0.7089 | 95.24 |
| Meta-Llama-3-8B-Instruct | 4 | 264.07 | 5.82 | 52.50 | 0.7180 | 96.74 |
| Qwen3-30B-A3B | 1 | 14207.53 | 6.56 | 748.23 | 0.8704 | 209.93 |
| Qwen3-30B-A3B | 2 | 7018.25 | 6.36 | 696.65 | 0.8810 | 205.89 |
| Qwen3-30B-A3B | 4 | 3694.46 | 6.36 | 723.05 | 0.8832 | 217.62 |