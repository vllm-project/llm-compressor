# HIGGS Bitwidth Sweep: NVFP4A16 + FP8_DYNAMIC

Sweep of target average bitwidth from 4.0 to 8.0 using the model-free
convert_checkpoint path (higgs_nvfp4a16_fp8_model_free.py), plus uniform
baselines at 4-bit, 8-bit, and 16-bit.

## Configuration

- **Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
- **Schemes**: NVFP4A16 (4-bit weight-only) + FP8_DYNAMIC (8-bit dynamic)
- **Method**: convert_checkpoint + HiggsQuantizationConverter (data-free)
- **Eval**: lm_eval wikitext word_perplexity via vLLM

## Results

| Target Bits | NVFP4A16 Layers | FP8_DYNAMIC Layers | word_perplexity |
|-------------|-----------------|---------------------|-----------------|
| 4.0 | 224 | 0 | 9.2765 |
| 4.5 | 124 | 100 | 9.1921 |
| 5.0 | 92 | 132 | 9.1225 |
| 5.5 | 75 | 149 | 9.0672 |
| 6.0 | 61 | 163 | 9.0381 |
| 6.5 | 46 | 178 | 9.0003 |
| 7.0 | 30 | 194 | 8.9685 |
| 7.5 | 15 | 209 | 8.9293 |
| 8.0 | 0 | 224 | 8.8891 |
| 16.0 (base) | 0 | 0 | 8.8187 |

Total layers: 224. HIGGS ILP solver allocates NVFP4A16 to the layers
with lowest sensitivity and FP8_DYNAMIC to the rest, respecting fused
group constraints (64 groups, 160 layers).

## Observations

- Base model (16-bit): 8.82 PPL. Uniform FP8_DYNAMIC (8-bit): 8.89 PPL —
  only 0.07 PPL degradation from full precision.
- Uniform NVFP4A16 (4-bit): 9.28 PPL — 0.46 PPL above base.
- HIGGS mixed-precision monotonically improves with more bits, from
  9.19 (4.5 bits) to 8.93 (7.5 bits).
- At 4.5 bits, HIGGS (9.19) already beats uniform 4-bit (9.28) by
  upgrading the most sensitive layers to FP8, confirming the ILP
  allocation is effective even at very low average bitwidth.
- PPL scales smoothly with ~0.05 PPL per 0.5-bit step and no sharp cliffs.
- The full range spans 0.46 PPL (8.82 to 9.28) across 4x compression.
- All checkpoints were produced in ~22 seconds each (data-free path).
