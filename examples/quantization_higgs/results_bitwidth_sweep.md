# HIGGS Bitwidth Sweep: NVFP4A16 + FP8_DYNAMIC

Sweep of target average bitwidth from 4.5 to 7.5 using the model-free
convert_checkpoint path (higgs_nvfp4a16_fp8_model_free.py).

## Configuration

- **Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
- **Schemes**: NVFP4A16 (4-bit weight-only) + FP8_DYNAMIC (8-bit dynamic)
- **Method**: convert_checkpoint + HiggsQuantizationConverter (data-free)
- **Eval**: lm_eval wikitext word_perplexity via vLLM

## Results

| Target Bits | NVFP4A16 Layers | FP8_DYNAMIC Layers | word_perplexity |
|-------------|-----------------|---------------------|-----------------|
| 4.5 | 124 | 100 | 9.4118 |
| 5.0 | 92 | 132 | 9.3373 |
| 5.5 | 75 | 149 | 9.2847 |
| 6.0 | 61 | 163 | 9.2471 |
| 6.5 | 46 | 178 | 9.2123 |
| 7.0 | 30 | 194 | 9.1760 |
| 7.5 | 15 | 209 | 9.1400 |

Total layers: 224. HIGGS ILP solver allocates NVFP4A16 to the layers
with lowest sensitivity and FP8_DYNAMIC to the rest, respecting fused
group constraints (64 groups, 160 layers).

## Observations

- PPL scales smoothly from 9.41 (4.5 bits) to 9.14 (7.5 bits), a 0.27 PPL
  range across 3 bits of average precision.
- The marginal cost of lower bitwidth is roughly linear: ~0.05 PPL per
  0.5-bit step, with no sharp cliff or diminishing returns.
- At 7.5 avg bitwidth only 15 layers use NVFP4A16 — the ILP is choosing
  the least sensitive layers to save bits.
- All checkpoints were produced in ~22 seconds each (data-free path).
