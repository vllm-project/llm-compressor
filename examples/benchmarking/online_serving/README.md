# Online Serving Benchmarking

When evaluating LLM performance for online serving, there are two latency metrics to consider:
- `TTFT` (Time to first token) measures how long it takes to generate the first token. 
- `TPOT` (Time per output token) measures how long it takes to generate each incremental token.

## Speedups from Weight-Only Quantization

On an Nvidia A10G GPU, we measure the following for online serving with the `sharegpt` dataset at 1 QPS:

| Model Stub                                | Precision     | TTFT (ms)     | TPOT (ms)     | Speedup vs Fp16   |
|-                                          |-              |-          |-              |-                  |
|`meta-llama/Meta-Llama-3-8B-Instruct`      |`fp16`         | 
|`neuralmagic/Meta-Llama-3-8B-Instruct-FP8` |`fp8`          |
|`astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit`|`int8`         |
|`nm-testing/Meta-Llama-3-8B-Instruct-GPTQ` |`int4`         |


## Generate Raw Benchmark Data

We can measure online serving latency by spinning up a vLLM server and creating sample clients. We will use `meta-llama/Meta-Llama-3-8B-Instruct` as the sample model:

```bash
export MODEL=meta-llama/Meta-Llama-3-8B-Instruct
```

### Spin Up vLLM Server

Install:

```bash
python -m venv vllm-venv
source vllm-venv/bin/activate
pip install vllm
```

Launch:

```bash
python -m vllm.entrypoints.openai.api_server --model $MODEL
```

### Spin Up Clients

Install:

```bash
python3 -m venv benchmark-venv
source benchmark-venv/bin/activate
pip install -U aiohttp transformers
```

Download sample data:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Launch the clients (we launch 1 client per second here):

```bash
python3 benchmark_serving.py \
    --model $MODEL \
    --request-rate 1.0 \
    --num-prompts 100 \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```

Results:

We achieve `43ms` of TPOT on an A10.

```bash
============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  110.95    
Total input tokens:                      22805     
Total generated tokens:                  17981     
Request throughput (req/s):              0.90      
Input token throughput (tok/s):          205.53    
Output token throughput (tok/s):         162.06    
---------------Time to First Token----------------
Mean TTFT (ms):                          106.36    
Median TTFT (ms):                        78.82     
P99 TTFT (ms):                           286.84    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          42.52     
Median TPOT (ms):                        43.07     
P99 TPOT (ms):                           58.80     
==================================================
```
