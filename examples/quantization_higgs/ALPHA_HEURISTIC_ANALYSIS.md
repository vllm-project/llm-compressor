# Alpha Heuristic Analysis for HIGGS Mixed-Precision Quantization

## Overview

This document summarizes the empirical analysis of per-layer quantization sensitivity
across multiple LLM architectures. The goal is to derive a heuristic alpha formula
that predicts how much each layer's quantization error (MSE) contributes to overall
perplexity degradation, enabling ILP-based optimal bitwidth allocation without
expensive per-model calibration.

The HIGGS paper models expected perplexity as:

    E[PPL(W_q)] ~ PPL(W) + sum_l( alpha_l * MSE_l )

where `alpha_l` is the per-layer importance weight. This analysis fits the form:

    alpha_l = base_func(size; type) * depth_factor(d) * scale

with:
- `base_func`: a function of layer parameter count, selected per layer type
- `depth_factor(d) = 1 + c1*d + c2*d^2`: a quadratic in layer depth
- `scale`: a global multiplier (absorbed into the ILP objective, does not affect allocation)

## Methodology

### Data Collection

1. **Bitwidth sweep**: For each model, quantize at 15 target average bitwidths
   (4.01 to 7.5 in 0.25 steps) using ILP with W4A16/W8A16 candidate schemes.
2. **MSE extraction**: Record per-layer MSE for each quantization scheme directly
   from the model safetensors (no calibration data needed).
3. **Perplexity evaluation**: Measure wikitext word_perplexity via lm_eval + vLLM
   for each quantized checkpoint and the baseline model.
4. **Regression fitting**: Grid search over 6 candidate base functions per layer type,
   fit depth quadratic coefficients via SLSQP with non-negativity constraints.

### Base Function Candidates

Six candidates are evaluated for each layer type:
- `log(s+1)` where s = parameter count
- `s` (linear)
- `s^2` (quadratic)
- `s^(-1)` (inverse)
- `sqrt(s)` (square root)
- `1` (constant / size-independent)

### Depth Curve Constraints

The depth quadratic `f(d) = 1 + c1*d + c2*d^2` is constrained to be non-negative
for all valid depths d in [0, max_depth]. This is enforced by checking:
- `f(0) >= 0` (always 1.0, trivially satisfied)
- `f(max_depth) >= 0`
- `f(-c1/(2*c2)) >= 0` if the vertex falls within [0, max_depth]

### Layer Type Classification

Layers are classified into types based on name patterns:
- **attention**: contains `self_attn` (q/k/v/o_proj)
- **mlp**: contains `mlp` but not `experts` or `shared_expert`
- **moe**: contains `experts.\d+.` (MoE expert layers)
- **moe-shared-expert**: contains `shared_expert`
- **embedding**: contains `embed`
- **default**: anything else (e.g., Gemma4's `per_layer_projection`)

## Models Evaluated

| Model | Params | Architecture | Layers | Quantizable Layers | Layer Types | Eval Points |
|-------|--------|-------------|--------|-------------------|-------------|-------------|
| Llama-3-8B | 8B | Dense | 32 | 224 | attention, mlp | 15/15 |
| Llama-3.1-8B | 8B | Dense | 32 | 224 | attention, mlp | 15/15 |
| Qwen2.5-7B | 7B | Dense | 28 | 196 | attention, mlp | 15/15 |
| Mistral-7B | 7B | Dense | 32 | 224 | attention, mlp | 15/15 |
| Qwen3-30B-A3B | 30B | MoE (128 experts) | 48 | 18624 | attention, moe | 8/15 |
| Gemma-4-E4B | 12B | Multimodal | 42 | 379 | attention, mlp, default | 0/15 |

Notes:
- Qwen3-30B-A3B has no shared experts; all MLP layers are routed expert layers.
- Gemma-4-E4B evaluation failed: baseline PPL=251.84 (instruction-tuned multimodal model)
  and all quantized checkpoints produce NaN in vLLM inference (compatibility issue).
- Qwen3-30B-A3B intermediate bitwidths (0.25-step) fail on vLLM with TP>=4
  (engine initialization failure); only the original 0.5-step evals succeeded.

## Fitting Results

### Best Fit Per Model

| Model | Residual | Base(attention) | Base(mlp/moe) | Scale |
|-------|----------|-----------------|---------------|-------|
| Mistral-7B | **3.07e-4** | sqrt(s) | sqrt(s) | 4.28 |
| Llama-3.1-8B | 2.00e-3 | 1 | log(s+1) | 2.59e2 |
| Llama-3-8B | 2.38e-3 | 1 | log(s+1) | 4.46e2 |
| Qwen3-30B-A3B | 4.12e-3 | log(s+1) | 1 | 4.51 |
| Qwen2.5-7B | 1.63e-2 | log(s+1) | log(s+1) | 1.02e2 |

Mistral-7B has the best fit by a wide margin, suggesting its PPL response to
quantization is the most predictable from MSE alone.

### Raw Depth Coefficients

| Model | Type | c1 | c2 | Max Depth |
|-------|------|-----|-----|-----------|
| Llama-3-8B | attention | +0.054884 | -0.002811 | 31 |
| Llama-3-8B | mlp | -0.093039 | +0.002972 | 31 |
| Llama-3.1-8B | attention | +0.201046 | -0.007526 | 31 |
| Llama-3.1-8B | mlp | -0.076475 | +0.002589 | 31 |
| Qwen2.5-7B | attention | -0.096514 | +0.002536 | 27 |
| Qwen2.5-7B | mlp | -0.151138 | +0.007692 | 27 |
| Mistral-7B | attention | -0.058253 | +0.001573 | 31 |
| Mistral-7B | mlp | -0.077680 | +0.002658 | 31 |
| Qwen3-30B-A3B | attention | -0.050264 | +0.000632 | 47 |
| Qwen3-30B-A3B | moe | -0.091133 | +0.002076 | 47 |

### Normalized Depth Curves

To compare across models with different layer counts, we normalize depth to
`t = d / max_depth` so t ranges from 0 (first layer) to 1 (last layer):

    f(t) = 1 + a*t + b*t^2    where a = c1*D, b = c2*D^2

#### Normalized Attention Coefficients

| Model | a | b | f(0) | f(0.5) | f(1) | min | t_min |
|-------|-----|-----|------|--------|------|-----|-------|
| Llama-3-8B | +1.70 | -2.70 | 1.00 | 1.18 | 0.00 | 0.00 | 1.00 |
| Llama-3.1-8B | +6.23 | -7.23 | 1.00 | 2.31 | 0.00 | 0.00 | 1.00 |
| Qwen2.5-7B | -2.61 | +1.85 | 1.00 | 0.16 | 0.24 | 0.08 | 0.70 |
| Mistral-7B | -1.81 | +1.51 | 1.00 | 0.48 | 0.71 | 0.46 | 0.60 |
| Qwen3-30B-A3B | -2.36 | +1.40 | 1.00 | 0.17 | 0.03 | 0.00 | 0.85 |

#### Normalized MLP/MoE Coefficients

| Model | Type | a | b | f(0) | f(0.5) | f(1) | min | t_min |
|-------|------|-----|-----|------|--------|------|-----|-------|
| Llama-3-8B | mlp | -2.88 | +2.86 | 1.00 | 0.27 | 0.97 | 0.27 | 0.51 |
| Llama-3.1-8B | mlp | -2.37 | +2.49 | 1.00 | 0.44 | 1.12 | 0.44 | 0.48 |
| Qwen2.5-7B | mlp | -4.08 | +5.61 | 1.00 | 0.36 | 2.53 | 0.26 | 0.36 |
| Mistral-7B | mlp | -2.41 | +2.55 | 1.00 | 0.43 | 1.15 | 0.43 | 0.47 |
| Qwen3-30B-A3B | moe | -4.28 | +4.59 | 1.00 | 0.00 | 1.30 | 0.00 | 0.47 |

## Key Findings

### 1. MLP/MoE Depth Curves Are Universal

All five models show the same qualitative MLP/MoE depth behavior:
- **U-shaped curve**: sensitivity starts at 1.0, drops to a minimum near the middle
  of the network (t ~ 0.4-0.5), then rises back.
- **Middle layers are least sensitive**: the minimum is consistently around t=0.47
  for most models (range 0.36-0.51).
- **Normalized coefficients cluster**: a in [-4.3, -2.4], b in [+2.5, +5.6].

This means MLP layers in the middle of any transformer can tolerate more aggressive
quantization, while the first and last MLP layers need higher precision.

A reasonable universal MLP heuristic:

    f_mlp(t) ~ 1 - 2.8*t + 2.8*t^2    (minimum ~0.30 at t=0.50)

### 2. Attention Depth Curves Split Into Two Families

**Family A - Llama (positive c1, negative c2):**
- Sensitivity RISES from layer 0, peaks in the upper-middle layers, then drops
  toward zero at the last layer.
- The network is most sensitive to attention quantization in layers around t=0.3-0.5.
- Last-layer attention can be quantized aggressively.

**Family B - Qwen/Mistral/Qwen3 (negative c1, positive c2):**
- Classic U-shape similar to MLP: sensitivity drops from layer 0, hits minimum
  around t=0.6-0.85, then recovers slightly.
- First-layer attention is the most sensitive.
- The minimum is deeper than MLP (as low as 0.00-0.08 for Qwen models).

This split may reflect architectural differences in how attention layers interact
with residual connections and layer normalization across model families.

### 3. Base Function Selection

- **`log(s+1)`** is the most commonly selected base function (wins for 3/5 models
  in at least one type), suggesting layer sensitivity scales sub-linearly with size.
- **`1` (constant)** wins for Llama attention, meaning attention sensitivity in
  Llama models is independent of layer size (all attention layers have similar sizes).
- **`sqrt(s)`** wins for Mistral, which may reflect its use of grouped-query attention
  (different Q vs KV sizes).
- The scale factor varies by orders of magnitude depending on the base function,
  but this does not affect ILP allocation (it cancels out in the objective).

### 4. Prediction Accuracy

Per-bitwidth PPL prediction error (predicted minus actual delta-PPL):

| Model | Max Error | Mean |Error| at 4-bit | at 7.5-bit |
|-------|-----------|-------------|--------|-----------|
| Mistral-7B | 0.011 | 0.004 | +0.00 | -0.011 |
| Llama-3.1-8B | 0.025 | 0.009 | +0.01 | -0.005 |
| Llama-3-8B | 0.026 | 0.010 | +0.01 | -0.012 |
| Qwen3-30B-A3B | 0.040 | 0.021 | -0.00 | -0.017 |
| Qwen2.5-7B | 0.077 | 0.024 | -0.00 | +0.021 |

For dense models, the heuristic predicts PPL within ~0.03 across the full 4-8 bit
range. Qwen2.5-7B has the largest errors, possibly due to non-smooth PPL response
or an imperfect base function choice.

## Recommendations for Default Heuristic

Based on these findings, a reasonable default alpha heuristic for unknown models:

```
alpha_l = base_func(size; type) * (1 + c1*d + c2*d^2) * scale

where:
  type = "attention" or "mlp" (classified from layer name)
  d    = layer depth index (0-based)
  D    = max layer depth
  t    = d / D  (normalized depth)

  For MLP layers:
    base_func = log(size + 1)
    Normalized: f(t) = 1 - 2.8*t + 2.8*t^2
    Raw: c1 = -2.8/D, c2 = 2.8/D^2

  For attention layers:
    base_func = log(size + 1)
    Normalized: f(t) = 1 - 2.0*t + 1.5*t^2
    Raw: c1 = -2.0/D, c2 = 1.5/D^2

  For MoE expert layers:
    base_func = 1  (experts have uniform size)
    Normalized: f(t) = 1 - 4.3*t + 4.6*t^2
    Raw: c1 = -4.3/D, c2 = 4.6/D^2
```

The attention default uses Family B coefficients (U-shape) as a conservative choice:
it prioritizes early layers, which is safe even for Llama-family models where early
attention sensitivity is moderate. The Llama-specific inverted curve could be used as
a model-specific override.

## Activation Quantization Analysis

### Experiment Design

To determine whether dynamic INT activation quantization affects quality (and thus
whether the ILP should include an activation bitwidth budget), we measured wikitext-2
perplexity across all HIGGS bitwidth allocations with activation quantization variants.

For each of the 6 Llama-3-8B HIGGS checkpoints (4.5-7.0 bit average), we created
symlink-farm variants with modified `config.json` that add dynamic INT8 or INT4
per-token activation quantization. Since WNaM weight tensors are identical regardless
of activation format, only the `config.json` differs — vLLM selects the appropriate
Marlin kernel (WNA4Int or WNA8Int) at runtime.

Variants tested:
- **A16**: Original (no activation quantization)
- **all-A8**: INT8 dynamic per-token activations on all layers
- **all-A4**: INT4 dynamic per-token activations on all layers
- **w4-A8**: INT8 activations only on W4 layers (W8 layers keep A16)

Cross-model validation: Llama-3.1-8B at 3 bitwidths × {A16, all-A8}.

### Results (Llama-3-8B, vLLM 0.26.1)

| Bitwidth | A16 | all-A8 | all-A4 | w4-A8 | Delta(A8) | Delta(A4) |
|----------|--------|--------|--------|--------|-----------|-----------|
| 4.5 | 9.0162 | 9.0159 | 9.0157 | 9.0157 | -0.0003 | -0.0005 |
| 5.0 | 8.8581 | 8.8583 | 8.8580 | 8.8588 | +0.0002 | -0.0001 |
| 5.5 | 8.6869 | 8.6864 | 8.6868 | 8.6874 | -0.0005 | -0.0001 |
| 6.0 | 8.6136 | 8.6141 | 8.6142 | 8.6135 | +0.0005 | +0.0006 |
| 6.5 | 8.5338 | 8.5332 | 8.5332 | 8.5330 | -0.0006 | -0.0006 |
| 7.0 | 8.4238 | 8.4241 | 8.4232 | 8.4240 | +0.0003 | -0.0006 |

### Cross-Model Validation (Llama-3.1-8B)

| Bitwidth | A16 | all-A8 | Delta(A8) |
|----------|--------|--------|-----------|
| 5.0 | 8.8952 | 8.8951 | -0.0001 |
| 6.0 | 8.6644 | 8.6647 | +0.0003 |
| 7.0 | 8.4869 | 8.4877 | +0.0008 |

### Conclusion

Dynamic INT activation quantization (both A8 and A4) has **zero measurable
perplexity impact** across all tested configurations. The maximum observed delta
is +/- 0.0008, well within measurement noise. This holds for:
- All weight bitwidth levels (4.5 to 7.0)
- Both INT8 and INT4 activations
- Both full-model and partial (W4-only) activation quantization
- Two different Llama architectures

**Implication for HIGGS:** The activation bitwidth budget constraint in the ILP is
purely a kernel selection knob (choosing between WNA4Int, WNA8Int, or WNA16 Marlin
kernels for inference speed), not a quality constraint. The heuristic coefficient
for activation quantization is effectively 0.

## Open Issues

1. **Gemma-4-E4B**: vLLM produces NaN for all quantized checkpoints (pack-quantized
   format incompatibility with Gemma4ForConditionalGeneration). Baseline PPL=251.84
   due to instruction-tuned multimodal model evaluated on wikitext. Needs either a
   vLLM fix or alternative evaluation method.

2. **Qwen3-30B-A3B intermediate evals**: The 0.25-step bitwidth checkpoints (4.25,
   4.75, 5.25, ...) fail to load in vLLM with TP>=4 (WorkerProc initialization
   failure). The 0.5-step originals work fine. May be related to uneven shard sizes
   caused by mixed-precision weight redistribution.

3. **Llama attention anomaly**: The Llama-family attention curves are qualitatively
   different from all other models (sensitivity rises then falls vs U-shape). More
   Llama-family models (e.g., Llama-3.1-70B, CodeLlama) would help confirm whether
   this is a Llama-specific pattern or an artifact of the 8B scale.

4. **Qwen2.5-7B fit quality**: Worst residual among tested models (1.63e-2). May
   benefit from a different functional form (e.g., piecewise linear) or additional
   base function candidates.
