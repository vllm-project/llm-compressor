# Greedy Multi-Scale Decomposition Guide

## Overview

Greedy Multi-Scale Decomposition implements a cascaded compression approach:

```
y = MPO_1(x) + LR_1(x) + MPO_2(x) + LR_2(x) + ...
```

**Key Idea:** Instead of fitting one massive Tensor-Train, build a "ladder" of approximations where each stage captures residual information missed by previous stages.

## Implementation

### Files Created

1. **`src/llmcompressor/modifiers/experimental/greedy_multiscale_linear.py`**
   - `GreedyMultiScaleLinear`: Main cascade layer
   - `LowRankLinear`: Low-rank correction layer
   - Iterative fitting: MPO → LR → MPO → LR → ...

2. **`test_greedy_multiscale.py`**
   - CPU test script
   - Tests multiple configurations
   - Validates on q/k/v/o projection layers

3. **`oneshot_greedy_multiscale.py`**
   - GPU-ready oneshot compression script
   - Integrates with calibration dataset
   - Uses real activations for fitting

## Initial Results (CPU Test)

### Configuration 1: MPO rank=0.3, LR rank=64

| Layer | Stages | Activation SNR | Params | Ratio | Status |
|-------|--------|----------------|--------|-------|--------|
| q_proj | 10 | 21.48 dB | 7,618,640 | 1.82x | ❌ Below 30 dB |
| k_proj | 10 | 22.71 dB | 2,396,180 | 2.29x | ❌ Below 30 dB |
| v_proj | 10 | 21.45 dB | 2,396,180 | 2.29x | ❌ Below 30 dB |
| o_proj | 10 | 18.87 dB | 7,618,640 | 1.82x | ❌ Below 30 dB |

**Observation:** Reaches ~21 dB but requires parameter expansion. Each stage adds incremental SNR (~2-4 dB per MPO, ~1-2 dB per LR).

## Recommended Configurations for GPU Testing

### Strategy 1: Higher MPO Rank (More Aggressive Base)

Use larger MPO rank to capture more structure in each stage:

```python
TARGET_SNR_DB = 30.0
MAX_STAGES = 5
MPO_RANK = 0.5      # Increased from 0.3
LR_RANK = 128       # Increased from 64
```

**Expected:** Better base approximation, fewer stages needed.

### Strategy 2: Balanced Approach

Balance between MPO and LR contributions:

```python
TARGET_SNR_DB = 30.0
MAX_STAGES = 4
MPO_RANK = 0.4
LR_RANK = 96
```

### Strategy 3: Lower Target SNR (Compression Focus)

If 30 dB is too aggressive, target lower SNR with actual compression:

```python
TARGET_SNR_DB = 20.0   # More realistic
MAX_STAGES = 3
MPO_RANK = 0.3
LR_RANK = 64
```

**Expected:** Likely achieves < 1.0x params with decent quality.

### Strategy 4: Layer-Specific Tuning

Different layers have different compressibility:
- **k_proj, v_proj**: Better candidates (smaller, lower rank)
- **q_proj, o_proj**: Harder to compress

```python
# For k_proj, v_proj:
MPO_RANK = 0.4
LR_RANK = 96

# For q_proj, o_proj:
MPO_RANK = 0.5
LR_RANK = 128
```

## How to Run on GPU

### Quick Test

```bash
python oneshot_greedy_multiscale.py
```

This will:
1. Load Llama-3.2-1B-Instruct on GPU
2. Collect activations from calibration data (32 samples)
3. Compress q/k/v/o projection layers
4. Save compressed model to `Llama-3.2-1B-Instruct-greedy-multiscale/`

### Customization

Edit the config section in `oneshot_greedy_multiscale.py`:

```python
# Compression settings
TARGET_SNR_DB = 30.0          # ← Adjust target SNR
MAX_STAGES = 5                # ← Max MPO+LR pairs
MPO_RANK = 0.3                # ← MPO rank (0.2-0.5)
LR_RANK = 64                  # ← LR rank (32-256)

# Layer targeting
COMPRESS_TARGETS = [
    "re:.*self_attn.(q|k|v|o)_proj$",  # Attention
    # "re:.*mlp.(gate|up|down)_proj$",  # Uncomment for MLP
]
```

### Monitor Progress

The script prints detailed progress:
- Activation collection per layer
- Each stage's SNR improvement
- Final compression stats
- Sample generation test

## Key Benefits

1. **Memory Efficient**: Small MPOs use less memory than one large MPO
2. **Numerically Stable**: Avoids barren plateaus of large tensor trains
3. **Adaptive**: Stops when target SNR reached
4. **Flexible**: Alternates between structured (MPO) and flat (LR) decompositions

## Potential Issues & Solutions

### Issue 1: Not Reaching Target SNR

**Symptoms:** Hits max_stages before target SNR

**Solutions:**
- Increase `MPO_RANK` (0.3 → 0.5)
- Increase `LR_RANK` (64 → 128)
- Increase `MAX_STAGES` (5 → 8)
- Lower `TARGET_SNR_DB` (30 → 25)

### Issue 2: Parameter Expansion

**Symptoms:** param_ratio > 1.0x

**Solutions:**
- Lower `TARGET_SNR_DB` (accept lower quality)
- Reduce `MAX_STAGES` (stop earlier)
- Reduce `LR_RANK` (smaller corrections)
- Focus on specific layers (k/v_proj compress better)

### Issue 3: Slow Inference

**Symptoms:** Compressed model runs slower than original

**Solutions:**
- Reduce number of stages (fewer operations)
- Consider "untensorizing" to dense matrix (trades memory for speed)
- Batch operations where possible

## Comparison to Other Approaches

| Approach | Params for 30 dB | Pros | Cons |
|----------|-----------------|------|------|
| **Pure SVD** | 1.76x (q_proj) | Optimal quality | Simple baseline |
| **Single MPO** | ~1.0x for 18 dB | Structured | Can't reach 30 dB |
| **Greedy Cascade** | ~1.5-2x for 30 dB? | Flexible, stable | More complex |

## Next Steps

1. **Run on GPU** with configurations above
2. **Test perplexity** - low SNR might be acceptable if perplexity is good
3. **Layer-specific tuning** - optimize k/v_proj separately from q/o_proj
4. **Activation-aware fitting** - use real activations (already implemented)
5. **Profile inference** - measure actual speedup/slowdown

## Expected Outcomes

**Optimistic:** Achieve 25-30 dB SNR with 0.8-1.2x parameters on k/v_proj

**Realistic:** Achieve 20-25 dB SNR with 0.6-0.8x parameters across all layers

**Conservative:** Need to accept parameter expansion (1.2-1.5x) for 30 dB SNR

---

**To run:** `python oneshot_greedy_multiscale.py` on a machine with GPU
