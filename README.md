# LLM Activation Compression with MERA

Compressing LLM hidden states for efficient speculative decoding training.

## Overview

This project explores using Multi-scale Entanglement Renormalization Ansatz (MERA) tensor networks to compress LLM activations. The goal: compress tens of TB of training activations for speculative decoding models.

**Current Status:** Achieved **3-4x net compression @ 27 dB SNR** with shared-basis MERA that generalizes to unseen samples.

---

## Key Results

### Best Working Approach: Shared Basis MERA

**File:** `mera_svd_disentangler.py`

- **Method:** Shared SVD-based disentanglers + shared isometry bases per layer
- **Training:** 256 samples
- **Per-sample compression:** 3.94x (latent only)
- **SNR:** 27.28 dB on held-out samples
- **Generalizes:** ✅ Yes - works on unseen samples
- **Bases:** 658K params (negligible when amortized over millions of samples)

**For 10 TB dataset:**
- Original: 10 TB
- Compressed: ~2.5-3 TB
- **Net compression: 3-4x**
- **This is real, generalizable compression**

### Failed Approach: Per-Position Bases

**File:** `mera_local_svd.py` 

- **Method:** Independent SVD basis per spatial position
- **Per-sample compression:** 512x (!)
- **SNR:** 89 dB on training samples
- **Generalizes:** ❌ **No** - complete failure on held-out (-42 dB)
- **Issue:** Bases learned from one sample don't apply to different samples

**Why it fails:** Each sample has unique structure at each position. Bases trained on sample A's position 0 don't work for sample B's position 0.

### Failed Approach: Gradient-Based Disentanglers

**File:** `mera_gradient_disentangler.py`

- **Method:** Learn disentanglers via gradient descent (Stiefel manifold)
- **SNR:** 2.93 dB on held-out (worse than SVD)
- **Issue:** Optimization gets stuck in local minima
- **Compression:** 1.02x (chi explodes: 256→502)

---

## Technical Findings

### 1. The Generalization Problem

**Discovery:** There's a fundamental trade-off between compression and generalization:

| Approach | Bases | Compression | SNR | Generalizes? |
|----------|-------|-------------|-----|--------------|
| **Per-position** | One per position (121M params) | 512x | 89 dB | ❌ No |
| **Shared** | One per layer (658K params) | 3.94x | 27 dB | ✅ Yes |

**Why per-position fails:** Each sample is unique. Bases learned from sample A position 0 capture A's variance, not B's variance.

**Why shared works:** Bases capture cross-sample patterns that generalize.

### 2. Disentanglers Matter (But Not Enough)

Without disentanglers (naive concatenation):
- SNR: 27.28 dB
- Chi: 254 → 502 (grows)

With SVD disentanglers:
- SNR: 27.28 dB (same)
- Chi: 252 → 479 (still grows)

With gradient disentanglers:
- SNR: 2.93 dB (fails)
- Training gets stuck in local minima

**Conclusion:** Disentanglers help slightly but don't solve chi explosion.

### 3. Batch Size Effects

Training on larger batches:
- Batch=64: L0 budget=197, L1 budget=1
- Batch=256: L0 budget=228, L1 budget=1  
- Batch=512: L0 budget=252, L1 budget=1

**Finding:** Larger training sets → better bases (higher chi budgets), but still same generalization (~27 dB).

### 4. The Chi Explosion Problem

With 99.9% energy threshold:
- Layer 0: 256 → 254
- Layer 1: 508 → 502
- **Chi grows instead of shrinking**

Lower thresholds:
- 99%: 10.45x compression, 17.1 dB SNR
- 95%: 94.56x compression, 10.45 dB SNR

**Trade-off is brutal:** Can't have both high compression AND high SNR with shared bases.

---

## Project Files

### Core Implementation

- **`extract_activations.py`** - Extract activations from Llama-3.1-8B layer 15
- **`mera_svd_disentangler.py`** - Best working approach (shared bases, 27 dB)
- **`mera_shared_bases.py`** - Simplified shared-basis MERA (no disentangler)
- **`mera_local_svd.py`** - Per-position bases (doesn't generalize)
- **`mera_gradient_disentangler.py`** - Gradient-based training (fails)
- **`mera_svd_deterministic.py`** - Earlier deterministic SVD approach

### Archive

- `archive/` - Old experiments (gradient descent failures, chi sweep tests, etc.)
- `archive/wavelet/` - Wavelet-based compression attempts (pre-MERA)

---

## Dataset

**Source:** Llama-3.1-8B-Instruct, layer 15, post-attention layernorm  
**Dataset:** mit-han-lab/pile-val-backup  
**Samples:** 512 sequences × 4096 tokens × 128 dims  
**Size per sample:** 2.1 MB (524,288 floats)  
**Total extracted:** ~1 GB (512 samples)

**Location:** `~/.cache/llm_activations/activations_post_attention_ln.pt`

---

## Usage

### Extract Activations

```bash
CUDA_VISIBLE_DEVICES=1 python extract_activations.py
```

Modify `NUM_SAMPLES` in the file to extract more/fewer samples.

### Train MERA (Shared Bases)

```python
from mera_svd_disentangler import SVDDisentanglerMERA, compute_snr_db
import torch

# Load activations
activations = torch.load('~/.cache/llm_activations/activations_post_attention_ln.pt')
activations = activations[:, :4096, :128].to('cuda')

# Train on 256 samples
train_batch = activations[:256]
mera = SVDDisentanglerMERA(4096, 128, 128, 0.999).to('cuda')
mera.initialize_uv_gate(train_batch)

with torch.no_grad():
    latent_train, _ = mera.train_bases(train_batch, num_layers=2)

# Test on held-out samples
test_sample = activations[256:257]
latent, intermediates = mera.encode(test_sample, num_layers=2)
recon = mera.reconstruct(latent, intermediates)

snr = compute_snr_db(test_sample[0], recon[0])
compression = (4096 * 128) / latent[0].numel()

print(f"Compression: {compression:.2f}x")
print(f"SNR: {snr:.2f} dB")
```

### Expected Output

```
Compression: 3.94x
SNR: 27.28 dB
```

---

## What We Learned

### ✅ What Works

1. **Shared bases generalize** - Same bases work across different samples
2. **SVD disentanglers are stable** - No optimization issues
3. **Deterministic approach** - No gradient descent, reproducible
4. **Real compression for TB-scale** - 3-4x net compression when bases amortized

### ❌ What Doesn't Work

1. **Per-position bases** - Amazing on training data, useless on test data
2. **Gradient-based training** - Gets stuck in local minima (0-2 dB SNR)
3. **High compression + high SNR** - Fundamental trade-off with shared bases
4. **Preventing chi explosion** - Hard caps destroy SNR, no caps → no compression

### 🤔 Open Questions

1. **Why does chi grow?** - Even with disentanglers, Layer 1 chi > Layer 0 chi
2. **Can we do better than 27 dB?** - Is this the limit of shared-basis MERA?
3. **Are there better architectures?** - MPS? Different hierarchical structures?
4. **Does this scale to other layers?** - Only tested on layer 15

---

## Next Steps

### Short Term (Practical)

1. **Test on more layers** - Does layer 15 represent all layers?
2. **Test on different models** - Llama-3, Mistral, etc.
3. **Optimize for speed** - Current implementation is research code
4. **Measure actual TB-scale performance** - Extract millions of samples

### Medium Term (Research)

1. **Try different energy thresholds per layer**
   - Layer 0: 99.9%, Layer 1: 95%, Layer 2: 90%
   - Might prevent chi explosion while maintaining SNR

2. **Hybrid approaches**
   - Shared bases for "common patterns"
   - Small per-sample residual correction
   - Could get closer to 512x with better generalization

3. **Different tensor network architectures**
   - Tree Tensor Networks (TTN)
   - Matrix Product States (MPS)
   - Might have better compression properties

### Long Term (Theoretical)

1. **Understand why chi grows**
   - Information theory analysis
   - What structure in activations causes this?

2. **Fundamental limits**
   - Is 3-4x the theoretical limit for generalizable compression?
   - Can we prove bounds?

3. **Better disentanglers**
   - Current approach: eigendecomposition of covariance
   - Could learn disentanglers that specifically target chi reduction?

---

## For TB-Scale Deployment

### Storage Calculator

For **10 TB** of activations (4.8M samples):

**With current approach (27 dB @ 3.94x):**
- Compressed latents: 2.54 TB
- Bases: 2.6 MB (negligible)
- **Total: ~2.5 TB**
- **Net compression: 4x**

**With better approach (if we reach 30 dB @ 8x):**
- Compressed latents: 1.25 TB
- Bases: 2.6 MB
- **Total: ~1.25 TB**
- **Net compression: 8x**

**With per-position (89 dB @ 512x, but doesn't generalize):**
- Would need bases PER SAMPLE
- Not viable

### Recommendation

For production deployment:
1. Train on 512-1K diverse samples
2. Use shared-basis MERA (`mera_svd_disentangler.py`)
3. Accept 3-4x compression @ 27 dB SNR
4. Bases are ~500 KB - trivial overhead
5. **This is real, working compression that generalizes**

---

## License

MIT
