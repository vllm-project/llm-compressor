# Compression Approaches Summary

Testing on meta-llama/Llama-3.2-1B-Instruct q_proj layer (2048×2048)
Target: 30-40 dB SQNR with parameter reduction

## Results

### 1. Block-Diagonal (Concatenation)
- **Result**: 6 dB @ 0.75x params
- **Status**: ❌ Poor SNR
- **Issue**: Doesn't capture cross-feature interactions

### 2. Single Low-Rank (SVD)
- **Result**: 14.86 dB @ 0.60x params
- **Status**: ❌ Poor SNR
- **Issue**: Dense matrices need high rank for good approximation

### 3. Stacked Low-Rank (2 layers)
- **Result**: 70 dB @ 1.20x params
- **Status**: ❌ No compression
- **Issue**: Second layer needed to correct first adds too many params

### 4. Low-Rank + All-at-once Sparse
- rank=1228: 30 dB @ 1.55x params
- rank=614: 31 dB @ 1.30x params
- **Status**: ❌ No compression
- **Issue**: Sparse correction is too dense

### 5. Iterative Sparse (Magnitude-based)
- **Result**: 31.5 dB @ 0.85x params
- **Status**: ✅ 15% reduction
- **Method**: Iteratively select top 5% elements by magnitude
- **Issue**: High sparsity (85%) means significant index overhead

### 6. OLS Column Selection
- **Result**: 70 dB @ 0.50x params
- **Status**: ✅✅ 50% reduction, excellent SNR
- **Method**: Iteratively select input columns via correlation with residual
- **Advantage**: Block-sparsity (entire columns), less index overhead
- **Note**: 1025/2048 columns selected

### 7. Hybrid Low-Rank + Iterative Sparse
Best configuration (rank=204):
- **Result**: 30.5 dB @ 0.97x params
- **Status**: ❌ Minimal compression (2.6%)
- **Issue**: Sparse part still needs ~77% density to correct residual

## Recommendations

### For 30-40 dB SNR with compression:

**🥇 Best: OLS Column Selection**
- 50% parameter reduction
- 70 dB SNR (far exceeds target)
- Block-sparse structure (low index overhead)
- Implementation: Select ~50% of input columns via greedy residual correlation

**🥈 Alternative: Iterative Magnitude Sparse**
- 15% parameter reduction
- 31.5 dB SNR (meets target)
- Element-wise sparse (higher index overhead)
- Simpler to implement

### Key Insights

1. **Low-rank alone insufficient**: Real model weights have complex structure requiring rank approaching full dimension

2. **Sparse methods win**: Unstructured sparsity captures fine details better than low-rank for dense matrices

3. **Column-wise sparsity best**: OLS column selection achieves:
   - Better compression (50% vs 15%)
   - Better SNR (70 dB vs 31 dB)
   - Lower index overhead (block-sparse)

4. **Hybrid doesn't help**: Low-rank base + sparse residual totals to near-original parameter count

5. **Activation-aware selection crucial**: OLS (using input activations) dramatically outperforms magnitude-based selection

## Implementation Priority

1. **Implement OLS column selection** as primary compression method
2. Consider element-wise sparse as fallback for different use cases
3. Skip hybrid and pure low-rank approaches for dense layers

## Next Steps

- Implement OLS column selection in main codebase
- Test on multiple layers (q_proj, k_proj, v_proj, mlp)
- Benchmark inference speed with sparse kernels
- Compare against existing TensorizedLinear (Tensor Train decomposition)
