# MERA Compression: Troubleshooting Guide

## Current Status

**PROBLEM:** Cannot achieve both high compression AND high SNR simultaneously with shared-basis MERA.

**Working code:** `standalone_mera_demo.py` - demonstrates the fundamental trade-off

---

## The Fundamental Issue

### Chi Explosion

MERA compresses spatially (4096 → 2048 → 1024 → 512 = 8x) but chi (bond dimension) grows at each layer, canceling out the gains:

| Threshold | Spatial | Chi Growth | Net Compression | SNR |
|-----------|---------|------------|-----------------|-----|
| 99.9% | 8x reduction | 128→254→502→990 (7.7x growth) | **1.03x** | **29 dB** |
| 99% | 8x reduction | 128→230→409→720 (5.6x growth) | **1.42x** | 19 dB |
| 98% | 8x reduction | 128→207→329→509 (4x growth) | **2.01x** | 16 dB |
| 95% | 8x reduction | 128→150→163→166 (1.3x growth) | **6.17x** | 12 dB |

**Why chi grows:** Concatenating even/odd positions doubles the dimension. To preserve 99.9% energy, we need to keep almost all dimensions → chi ≈ 2× each layer.

---

## What We Tried

### 1. Per-Position Bases (Failed - Doesn't Generalize)

**Results:**
- Training samples: 512x compression @ 89 dB SNR
- Held-out samples: 2x compression @ **-42 dB SNR**

**Why it failed:** Each sample has unique structure. Bases learned from sample A don't work on sample B.

### 2. Shared Bases (Works but No Compression)

**Results:**
- **1.03x compression @ 29 dB SNR** with 99.9% threshold
- Generalizes perfectly ✓
- But no compression ✗

### 3. Lower Energy Thresholds

| Threshold | Compression | SNR | Issue |
|-----------|-------------|-----|-------|
| 99.9% | 1.03x | 29 dB | No compression |
| 98% | 2.01x | 16 dB | SNR too low? |
| 95% | 6.17x | 12 dB | SNR unusable |

---

## Critical Question

**Is 15-20 dB SNR acceptable for speculative decoding training?**

- If yes → Use 98% threshold, get 2x compression
- If no → Need different approach (not shared-basis MERA)

---

## How to Reproduce

```bash
python standalone_mera_demo.py
```

Expected output:
```
Threshold       Chi   Compress        SNR
  0.999  254→502→990      1.03x     29dB  (no compression)
  0.980  207→329→509      2.01x     16dB  (some compression)
  0.950  150→163→166      6.17x     12dB  (too noisy?)
```

---

## Files

- `standalone_mera_demo.py` - Self-contained demo
- `mera_svd_disentangler.py` - Full implementation
- `test_threshold_sweep.py` - Comprehensive tests
- `README.md` - Project overview
- `SESSION_SUMMARY.md` - What we tried

