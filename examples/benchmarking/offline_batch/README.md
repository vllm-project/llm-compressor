# Offline Batch Benchmarking

When evaluating LLM performance for online serving, we should focus on throughput metrics such as tokens/second or requests/second.

## Speedups from Activation Quantization

On an Nvidia A10G GPU, we measure the following for offline batch processing:

| Model Stub                                | Precision     | TTFT (ms)     | TPOT (ms)     | Speedup vs Fp16   |
|-                                          |-              |-          |-              |-                  |
|`meta-llama/Meta-Llama-3-8B-Instruct`      |`fp16`         | 
|`nm-testing/Meta-Llama-3-8B-Instruct-W8-Channel-A8-Dynamic-Per-Token-Test` |`int8`          |

## Generate Raw Benchmark Data

We can measure online serving latency by running vLLM on a sample dataset. We will use `meta-llama/Meta-Llama-3-8B-Instruct` as the sample model:

```bash
export MODEL=meta-llama/Meta-Llama-3-8B-Instruct
```

Install:

```bash
python -m venv vllm-venv
source vllm-venv/bin/activate
pip install vllm
```

Run sample workload:

```bash
python benchmark_offline.py --help
python benchmark_offline.py --model $MODEL
```

Results:

```bash
* ==========================================================
* Total Time:                   54.27
* Total Generations:            1000


* Generations / Sec:            18.43
* Generation Tok / Sec:         4198.43
* Prompt Tok / Sec:             10425.07


* Avg Generation Tokens:        227.85
* Avg Prompt Tokens:            565.78
* ==========================================================
```
