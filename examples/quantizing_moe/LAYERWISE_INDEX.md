# Layerwise Quantization Examples — Complete Index

New to layerwise decompression? Start here.

## 📚 Documentation Structure

```
quantizing_moe/
├── LAYERWISE_INDEX.md                    ← You are here
├── README_LAYERWISE.md                   ← Quick overview & file guide
├── LAYERWISE_DECOMPRESSION.md            ← Deep dive explanation
├── TESTING_LAYERWISE.md                  ← How to test & debug
│
├── kimi_k3_example.py                    ← Standard quantization (baseline)
├── kimi_k3_layerwise_example.py          ← Sequential pipeline (use now)
└── kimi_k3_layerwise_full_example.py     ← Full layerwise (coming soon)
```

## 🎯 Which Document Should I Read?

### I want a quick overview
→ **[README_LAYERWISE.md](README_LAYERWISE.md)** (5 min)
- What are the files?
- When to use each?
- Quick start commands
- Memory comparison table

### I want to understand the concept
→ **[LAYERWISE_DECOMPRESSION.md](LAYERWISE_DECOMPRESSION.md)** (15 min)
- What problem does it solve?
- How does layerwise work?
- Sequential vs. full layerwise
- Kimi-K3 architecture details
- References & further reading

### I want to test an example
→ **[TESTING_LAYERWISE.md](TESTING_LAYERWISE.md)** (20 min)
- Step-by-step testing guide
- Memory monitoring
- Validation checklist
- Troubleshooting guide
- Expected outputs

### I want to run code right now
→ **[kimi_k3_layerwise_example.py](kimi_k3_layerwise_example.py)** (Ready Now)
```bash
python kimi_k3_layerwise_example.py
```

### I want to see the future version
→ **[kimi_k3_layerwise_full_example.py](kimi_k3_layerwise_full_example.py)** (Coming Soon)
- Template for when flags are merged
- Shows explicit decompress/compress pattern
- Comments on what to expect

## 🚀 Quick Navigation

### For Different Use Cases

**Learning about layerwise:**
1. [README_LAYERWISE.md](README_LAYERWISE.md) — Overview
2. [LAYERWISE_DECOMPRESSION.md](LAYERWISE_DECOMPRESSION.md) — Deep dive
3. [kimi_k3_layerwise_example.py](kimi_k3_layerwise_example.py) — See code

**Running an example:**
1. [TESTING_LAYERWISE.md](TESTING_LAYERWISE.md) — Setup & run guide
2. [kimi_k3_layerwise_example.py](kimi_k3_layerwise_example.py) — Execute
3. Monitor with `nvidia-smi`

**Troubleshooting:**
1. [TESTING_LAYERWISE.md](TESTING_LAYERWISE.md) — Troubleshooting section
2. Check sequential pipeline docs
3. Monitor GPU memory & calibration progress

**Future features:**
1. [README_LAYERWISE.md](README_LAYERWISE.md) — Full Layerwise section
2. [LAYERWISE_DECOMPRESSION.md](LAYERWISE_DECOMPRESSION.md) — Coming Soon section
3. [kimi_k3_layerwise_full_example.py](kimi_k3_layerwise_full_example.py) — Template

## 📖 What Each File Does

### Example Scripts

| Script | Status | What It Does |
|--------|--------|-------------|
| `kimi_k3_example.py` | ✅ Available | Standard NVFP4 quantization, single-pass calibration |
| `kimi_k3_layerwise_example.py` | ✅ Ready Now | Sequential pipeline with per-layer processing |
| `kimi_k3_layerwise_full_example.py` | 🔮 Coming Soon | Explicit per-layer decompress→calibrate→compress |

### Documentation

| Doc | Purpose | Key Sections |
|-----|---------|--------------|
| `README_LAYERWISE.md` | Overview & quick start | Files, quick start, key differences, memory profile |
| `LAYERWISE_DECOMPRESSION.md` | Concept explanation | Problem, solution, how it works, Kimi-K3 specifics |
| `TESTING_LAYERWISE.md` | How to test | Quick start, memory monitoring, validation, troubleshooting |

## 🎓 Learning Path

### Path 1: I'm New to This (40 min total)
1. Read [README_LAYERWISE.md](README_LAYERWISE.md) — 5 min
2. Skim [LAYERWISE_DECOMPRESSION.md](LAYERWISE_DECOMPRESSION.md) — 10 min
3. Follow [TESTING_LAYERWISE.md](TESTING_LAYERWISE.md) — 25 min (run example)

**Outcome**: Understand layerwise, successfully run sequential pipeline example

### Path 2: I Know PyTorch, Want Details (60 min total)
1. Read [LAYERWISE_DECOMPRESSION.md](LAYERWISE_DECOMPRESSION.md) — 20 min
2. Read all example code — 15 min
3. Follow [TESTING_LAYERWISE.md](TESTING_LAYERWISE.md) — 25 min

**Outcome**: Deep understanding, ready to modify examples for your use case

### Path 3: I Just Want to Run It (15 min total)
1. Quick glance at [README_LAYERWISE.md](README_LAYERWISE.md) — 2 min
2. Follow "Quick Start" in [TESTING_LAYERWISE.md](TESTING_LAYERWISE.md) — 13 min

**Outcome**: Sequential example running on your hardware

## 🔑 Key Concepts Explained

### Sequential Pipeline
Process model layer-by-layer, keeping only one layer in GPU memory.
**File**: See examples in [kimi_k3_layerwise_example.py](kimi_k3_layerwise_example.py)
**Docs**: Explained in [README_LAYERWISE.md](README_LAYERWISE.md)

### Layerwise Decompression
Explicitly decompress one layer at a time before recalibration.
**File**: See template in [kimi_k3_layerwise_full_example.py](kimi_k3_layerwise_full_example.py)
**Docs**: Deep dive in [LAYERWISE_DECOMPRESSION.md](LAYERWISE_DECOMPRESSION.md)

### Error Propagation
Adjust next layer's inputs based on quantization errors from current layer.
**Where to read**: [LAYERWISE_DECOMPRESSION.md](LAYERWISE_DECOMPRESSION.md#layerwise-decompression-how-it-works)

## ⚙️ Parameter Reference

### Sequential Pipeline Parameters
```python
pipeline="sequential",
sequential_targets=["KimiK3DecoderLayer"],  # Layer class to process
propagate_error=True,                        # Adjust next layer
batch_size=1,                                # One sample per step
```
→ Explained in [README_LAYERWISE.md](README_LAYERWISE.md#key-differences)

### Layerwise Flags (Coming Soon)
```python
layerwise_decompression=True,   # Decompress before each layer
layerwise_compression=True,     # Recompress after each layer
```
→ Template in [kimi_k3_layerwise_full_example.py](kimi_k3_layerwise_full_example.py)
→ Will be documented in [TESTING_LAYERWISE.md](TESTING_LAYERWISE.md) once merged

### Quantization Ignore Patterns
```python
qconfig.ignore += [
    "re:.*mlp_res_proj.*",
    "re:.*self_attention_res_proj.*",
    "re:.*routed_expert.*",
    "re:.*output_attn_res_proj.*",
]
```
→ Explained in [LAYERWISE_DECOMPRESSION.md](LAYERWISE_DECOMPRESSION.md#kimi-k3-specifics)

## 🧪 Testing Overview

**Quick test (5 min):**
```bash
python kimi_k3_layerwise_example.py
```

**Full validation (30 min):**
1. Follow [TESTING_LAYERWISE.md](TESTING_LAYERWISE.md) — Quick Start section
2. Monitor memory with `nvidia-smi`
3. Validate model quality with test generation

**Compare approaches (1 hour):**
```bash
# Standard
time python kimi_k3_example.py
# Sequential
time python kimi_k3_layerwise_example.py
# Compare memory & speed
```

→ Full testing guide in [TESTING_LAYERWISE.md](TESTING_LAYERWISE.md)

## 📊 Memory Comparison

| Approach | Peak Memory | When to Use |
|----------|-------------|------------|
| Standard | 100+ GB | Small models, plenty of VRAM |
| Sequential | 60-70 GB | 100B+ models, limited VRAM |
| Full Layerwise | 40-50 GB | Explicit control needed |

→ See [README_LAYERWISE.md](README_LAYERWISE.md#memory-profile-comparison)

## 🐛 Troubleshooting Quick Links

| Problem | Solution |
|---------|----------|
| Model won't load | Check Kimi-K3 in HuggingFace, check `trust_remote_code=True` |
| OOM during calibration | See [TESTING_LAYERWISE.md](TESTING_LAYERWISE.md#oom-during-calibration) |
| Calibration hangs | See [TESTING_LAYERWISE.md](TESTING_LAYERWISE.md#calibration-hangs) |
| Bad model outputs | See [TESTING_LAYERWISE.md](TESTING_LAYERWISE.md#model-accuracy-issues) |
| Layerwise flags missing | See [kimi_k3_layerwise_full_example.py](kimi_k3_layerwise_full_example.py) — coming soon |

## 🔗 Related Resources

- **Kimi-K3 Model**: https://huggingface.co/moonshotai/Kimi-K3
- **LLM Compressor**: https://github.com/vllm-project/llm-compressor
- **PR #2994**: https://github.com/vllm-project/llm-compressor/pull/2994
- **Layerwise Commit**: `7e0a204cd` (in main branch)
- **Compressed-Tensors**: https://github.com/vllm-project/compressed-tensors

## 📝 Summary

This directory contains:
- ✅ **2 working examples** (standard + sequential)
- 🔮 **1 future template** (full layerwise, coming soon)
- 📖 **3 detailed guides** (overview, concept, testing)
- 🎯 **This index** to help you navigate

Start with [README_LAYERWISE.md](README_LAYERWISE.md) for a quick overview, then dive into the specific doc you need.

Good luck! 🚀
