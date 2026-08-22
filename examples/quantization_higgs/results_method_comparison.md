# HIGGS Method Comparison: Llama-3.1-8B-Instruct

Mixed-precision quantization at 6.0 average bitwidth using HIGGS ILP allocation.
All runs use 61 low-precision + 163 FP8_DYNAMIC layers (224 total).

## Configuration

- **Model**: meta-llama/Meta-Llama-3.1-8B-Instruct
- **Target bitwidth**: 6.0 avg
- **Calibration** (where applicable): ultrachat_200k, 256 samples, 2048 seq len
- **Eval**: lm_eval wikitext word_perplexity via vLLM

## Results

| Example | Schemes | Method | HIGGS Time | Quant Time | PPL |
|---------|---------|--------|------------|------------|-----|
| higgs_nvfp4_fp8_qmod | NVFP4 + FP8_DYNAMIC | QuantizationModifier | 3.5s | 1:21 | 9.3860 |
| higgs_nvfp4_fp8_gptq | NVFP4 + FP8_DYNAMIC | GPTQModifier | 3.5s | 10:28 | 9.3760 |
| higgs_nvfp4a16_fp8_model_free | NVFP4A16 + FP8_DYNAMIC | convert_checkpoint | 3.5s | 0:22 | 9.2483 |
| higgs_nvfp4a16_fp8_gptq | NVFP4A16 + FP8_DYNAMIC | GPTQModifier | 3.5s | 10:14 | 9.2493 |

## Observations

- **GPTQ provides negligible improvement** at 6.0 bitwidth — both GPTQ variants
  match their non-GPTQ counterparts within noise (0.01 PPL), at ~8x quant time.
- **NVFP4A16 outperforms NVFP4** (9.25 vs 9.39 PPL) at the same average bitwidth.
  FP8_DYNAMIC already handles activations well, so W4A4 offers no benefit over W4A16.
- **convert_checkpoint is the fastest path** — 22 seconds total, no model load or
  calibration needed, and identical quality to the GPTQ path for NVFP4A16.
