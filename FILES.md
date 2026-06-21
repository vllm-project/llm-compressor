# Project Files

## Documentation

- **README.md** - Main documentation, usage guide, results summary
- **SESSION_SUMMARY.md** - Detailed session notes, what we tried, what worked
- **RESULTS_VARIATIONAL_MERA.md** - Archived detailed experimental results

## Core Implementation (Keep These)

### Best Working Approach
- **mera_svd_disentangler.py** - Shared bases with SVD disentanglers (27 dB @ 3.94x)
  - Use this for production
  - Generalizes to unseen samples
  - Deterministic, no training required

### Alternative Approaches (For Research)
- **mera_shared_bases.py** - Simplified shared-basis MERA (no disentangler)
- **mera_svd_deterministic.py** - Early deterministic SVD implementation
- **mera_local_svd.py** - Per-position bases (doesn't generalize, archived for reference)
- **mera_gradient_disentangler.py** - Gradient-based training (fails, archived for reference)

### Data Extraction
- **extract_activations.py** - Extract activations from LLM
  - Currently set to 512 samples
  - Modify NUM_SAMPLES for more/less

## Archive

### archive/
Old experimental code:
- `binary_mera*.py` - Early gradient descent attempts (all failed)
- `test_*.py` - Various test scripts
- `mera_batch*.py` - Batch processing experiments  
- `mera_chi_sweep*.py` - Chi parameter sweeps
- `mera_hierarchical.py` - Alternative hierarchical approach
- `mera_adaptive_chi.py` - Attempted per-position chi (broken)
- Old documentation files

### archive/wavelet/
Pre-MERA compression attempts using wavelets:
- Various wavelet-based compression experiments
- MPS/MPO tensor network approaches
- Spectrum analysis tools

## Cached Data

Location: `~/.cache/llm_activations/`

- `activations_post_attention_ln.pt` - 512 samples, 1 GB
- `activations_post_input_ln.pt` - 512 samples, 1 GB

## Quick Start

### Run best approach:
```bash
python mera_svd_disentangler.py
```

### Extract more data:
```bash
CUDA_VISIBLE_DEVICES=1 python extract_activations.py
```

## File Size Guide

```
Core implementations:    ~60 KB (6 files)
Archive:                 ~500 KB (40+ files)
Documentation:           ~50 KB (3 files)
Cached activations:      ~2 GB
```

## What to Keep vs Archive

### Keep for production:
- `mera_svd_disentangler.py`
- `extract_activations.py`
- `README.md`

### Keep for research:
- `mera_shared_bases.py`
- `mera_svd_deterministic.py`
- `SESSION_SUMMARY.md`

### Archive (reference only):
- Everything in `archive/`
- All other mera_*.py files
