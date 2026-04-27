# KL Divergence Evaluation

Measures output distribution shift between a baseline and quantized model using
vLLM hidden state extraction.

## Quick start

```python
from llmcompressor.evaluation import evaluate_kl_divergence

result = evaluate_kl_divergence(
    base_model_id="meta-llama/Meta-Llama-3-8B",
    quantized_model_id="./Meta-Llama-3-8B-W4A16",
)
print(f"Mean KLD: {result.mean_kld:.6f}")
```

## How it works

Extracting full vocabulary logprobs from vLLM is slow (~64 hours for WikiText
on Llama-3-8B). This tool extracts **pre-`lm_head` hidden states** instead
(`hidden_dim ≈ 4096` vs `vocab_size ≈ 120k`), reducing data volume ~30x.

Pipeline:
1. Register a forward hook on `lm_head` via `LLM.apply_model`
2. Run inference on calibration prompts; capture hidden states on CPU
3. Apply the baseline `lm_head` to both hidden state sets offline
4. Compute `KL(P_base || P_quant)` per token, averaged over all prompts

## Colab notebook

[`kld_metric.ipynb`](kld_metric.ipynb) — runs end-to-end on a free T4 GPU:
quantizes `facebook/opt-125m` with W4A16 and evaluates KLD against the baseline.

## Interpreting results

| Mean KLD | Interpretation |
|----------|----------------|
| ~0.0     | Identical distributions (sanity check) |
| < 0.1    | Excellent — minimal quantization degradation |
| 0.1–1.0  | Moderate shift — review calibration or scheme |
| > 1.0    | Large shift — consider higher-precision quantization |
