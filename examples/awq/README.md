# MAI 2026 Efficient LLMs Challenge — Optimized On-Device Inference

## Architecture

```
mai2026_efficient_llm/
├── kernels/                    # C/NEON optimized compute kernels
│   ├── quantize.h              # Quantization data structures
│   ├── gemv_neon.c             # ARM NEON GEMV kernels (W4A8, W2A8, mixed)
│   ├── gemv_reference.c        # Reference C kernels (Colab/x86 fallback)
│   └── Makefile                # Cross-compile for ARM64 or native
├── engine/                     # Python inference engine
│   ├── __init__.py
│   ├── model_loader.py         # Load & quantize HF models
│   ├── quantizer.py            # Mixed-precision quantization with layer importance
│   ├── inference.py            # Token-by-token generation with custom kernels
│   └── benchmark.py            # Benchmarking utilities
├── configs/                    # Model & optimization configs
│   └── qwen2.5_0.5b.yaml
├── scripts/
│   ├── export_gguf.py          # Export to GGUF for llama.cpp comparison
│   └── deploy_pi.sh            # Pi 5 deployment script
├── colab_demo.ipynb            # Google Colab notebook (auto-generated)
├── run_colab.py                # Colab-compatible entry point
├── run_pi5.py                  # Pi 5 optimized entry point
└── requirements.txt
```

## Quick Start (Colab)
```bash
pip install -r requirements.txt
python run_colab.py --model Qwen/Qwen2.5-0.5B-Instruct --bits 4
```

## Quick Start (Pi 5)
```bash
cd kernels && make arm64
cd .. && python run_pi5.py --model Qwen/Qwen2.5-0.5B-Instruct --bits 4 --threads 4
```
