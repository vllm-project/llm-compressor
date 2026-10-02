# KL Divergence Evaluation

Measures output distribution shift between a baseline and quantized model using
vLLM hidden state extraction.

## Quick start

```python
from llmcompressor.evaluation import evaluate_kl_divergence

result = evaluate_kl_divergence(
    base_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    quantized_model_id="./Meta-Llama-3-8B-Instruct-W4A16",
    dataset="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    num_calibration_samples=512,
)
print(f"Mean KLD: {result.mean_kld:.6f}")
```

## CLI usage

```bash
python -m llmcompressor.evaluation.kld \
    --base_model_id meta-llama/Meta-Llama-3-8B-Instruct \
    --quantized_model_id ./Meta-Llama-3-8B-Instruct-W4A16 \
    --dataset wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --num_calibration_samples 512
```

All options:

| Flag | Default | Description |
|------|---------|-------------|
| `--base_model_id` | required | Baseline model HF ID or local path |
| `--quantized_model_id` | required | Quantized model HF ID or local path |
| `--dataset` | `wikitext` | HF dataset ID |
| `--dataset_config_name` | `wikitext-2-raw-v1` | HF dataset config |
| `--dataset_split` | `test` | Dataset split |
| `--text_column` | `text` | Column containing text |
| `--num_calibration_samples` | `512` | Number of prompts to evaluate |
| `--max_seq_length` | `512` | Max characters per prompt |
| `--dtype` | `auto` | vLLM model dtype |
| `--max_tokens` | `1` | Tokens to generate per prompt |
| `--temperature` | `0.0` | Sampling temperature |
| `--gpu_memory_utilization` | `0.45` | GPU memory per model (~0.90 total) |

## Example script

[`llama3_example.py`](llama3_example.py) — quantizes `meta-llama/Meta-Llama-3-8B-Instruct`
with W4A16 using WikiText-2 calibration data, then evaluates KLD against the baseline.

```bash
python examples/kld_evaluation/llama3_example.py
```

## How it works

Extracting full vocabulary logprobs from vLLM is slow (~64 hours for WikiText
on Llama-3-8B). This tool extracts **pre-`lm_head` hidden states** instead
(`hidden_dim ~= 4096` vs `vocab_size ~= 120k`), reducing data volume ~30x.

Pipeline:
1. Load both models concurrently (`gpu_memory_utilization=0.45` each)
2. Patch `LogitsProcessor.forward` via `LLM.apply_model` to capture pre-`lm_head`
   hidden states into a pre-allocated CUDA buffer (no `enforce_eager` needed --
   buffer writes are CUDA ops, CUDA-graph compatible)
3. Run inference per prompt on both models; retrieve hidden states to CPU online
4. Apply the baseline `lm_head` to both hidden state sets offline
5. Compute `KL(P_base || P_quant)` per token, averaged over all prompts

## Interpreting results

| Mean KLD | Interpretation |
|----------|----------------|
| ~0.0     | Identical distributions (sanity check) |
| < 0.1    | Excellent -- minimal quantization degradation |
| 0.1-1.0  | Moderate shift -- review calibration or scheme |
| > 1.0    | Large shift -- consider higher-precision quantization |
