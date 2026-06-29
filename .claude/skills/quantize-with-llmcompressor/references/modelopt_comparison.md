# LLM Compressor vs NVIDIA TensorRT Model Optimizer

NVIDIA's public skills catalog (`github.com/NVIDIA/skills`) ships product skills
(Jetson, cuOpt, NeMo, Holoscan, …) but **no standalone quantization / Model
Optimizer skill**. The closest comparable workflow is the
[TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
library (`nvidia-modelopt`). This page compares the two on the same task so the
skill can give honest guidance about when each is the better tool.

## Design / capability comparison

| Dimension | LLM Compressor | NVIDIA Model Optimizer |
|---|---|---|
| Project / license | vLLM project, Apache-2.0 | NVIDIA, Apache-2.0 |
| Primary deployment target | **vLLM** (native `compressed-tensors`) | TensorRT-LLM (also HF export → vLLM/SGLang for FP8/NVFP4) |
| Output format | `compressed-tensors` safetensors, `vllm serve`-ready as-is | Simulated quant in-graph → `export_hf_checkpoint` / TRT-LLM engine |
| One-call API | `oneshot(model, recipe, dataset)` | `mtq.quantize(model, config, forward_loop)` + export |
| Data-free PTQ | Yes (FP8 dynamic/block, W*A16, NVFP4A16) | Yes (FP8, etc.) |
| 4-bit weight algorithms | RTN, **GPTQ, AWQ, AutoRound** | AWQ, max/SmoothQuant calibrators |
| FP4 support | NVFP4 (W4A4 + W4A16) **and MXFP4/MXFP8** | NVFP4, W4A8; MXFP4 via TRT path |
| KV-cache / attention quant | Yes (per-head KV, attention FP8/NVFP4) | KV-cache FP8 |
| Beyond quantization | quantization + sparsity (legacy) | quantization + **pruning + distillation + speculative decoding + QAT** |
| Huge-model path | `model_free_ptq`, sequential onloading | layer-wise + TRT builder |
| Hardware-aware scheme routing | This skill (`detect_hardware.py`) | Manual config choice |

### When to prefer each

* **Choose LLM Compressor** when your serving stack is **vLLM** (or SGLang via
  compressed-tensors). The output loads directly with no engine build, the
  algorithm menu for 4-bit weights is richer (GPTQ/AWQ/AutoRound), and it covers
  microscale formats (MXFP4/MXFP8) and fine-grained KV/attention quant.
* **Choose Model Optimizer** when you deploy through **TensorRT-LLM** on NVIDIA
  hardware, or you need the broader optimization toolbox (QAT, pruning,
  distillation, speculative decoding) in one library.

## Empirical comparison (same model, same scheme)

Reproduce with:

```bash
python scripts/benchmark_compare.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --scheme FP8_DYNAMIC --tools both --json-out bench.json
```

It quantizes the *same* base model to the *same* target scheme with both tools
and reports WikiText-2 perplexity (accuracy), exported checkpoint size, and
quantization wall-time.

> **Heads-up — the two tools need separate virtualenvs.** `nvidia-modelopt`
> 0.44 pins `transformers<5.6`, while current `llmcompressor` needs
> `transformers>=5.10` (the `deepseek_v4` model). They cannot coexist in one
> environment. Run each side in its own venv and pass `--tools llmcompressor`
> in one and `--tools modelopt` in the other (each branch only imports the
> library it needs), then combine the two JSON outputs:
>
> ```bash
> .llmc-env/bin/python   scripts/benchmark_compare.py --tools llmcompressor \
>     --scheme FP8_DYNAMIC --json-out llmc.json
> .modelopt-env/bin/python scripts/benchmark_compare.py --tools modelopt \
>     --scheme FP8_DYNAMIC --json-out modelopt.json
> ```

<!-- RESULTS:START -->
Measured on an **RTX 5090 Laptop (Blackwell, compute 12.0, 24 GB)**, Windows 11,
torch 2.11.0+cu128, `Qwen/Qwen2.5-0.5B-Instruct`, WikiText-2 perplexity (40
windows), 128 calibration samples where calibration applies:

| Scheme | Tool | Perplexity ↓ | Size (GB) ↓ | Quant step | vLLM-ready |
|---|---|---|---|---|---|
| BF16 (baseline) | — | 12.23 | ~0.94 | — | n/a |
| FP8_DYNAMIC | **LLM Compressor** | 12.40 | 0.598 | data-free RTN (~6 s) | ✅ as-is |
| FP8 | NVIDIA Model Optimizer | 12.41 | 0.587 | calibrated (~30 s) | after export |
| NVFP4 | **LLM Compressor** | 14.84 | **0.452** | calibrated (~26 s) | ✅ as-is |
| NVFP4 | NVIDIA Model Optimizer | — | — | — | ❌ did not run¹ |

¹ ModelOpt's NVFP4/MX path tried to JIT-build a CUDA C++ extension
(`modelopt_cuda_ext_mx`) and failed because no MSVC (`cl.exe`) toolchain was
present (`Command '['where', 'cl']' returned non-zero exit status 1`). Its FP8
path also fell back to simulated quantization for the same reason but still
completed. LLM Compressor's FP4 path is `torch.compile`-decorated and needs
Triton; the skill's scripts auto-fall-back to eager (`TORCHDYNAMO_DISABLE=1`)
when Triton is absent, so NVFP4 ran to completion on the same machine.

**Reading these results:**

* **FP8 is a dead heat on accuracy** (12.41 vs 12.40, both within ~1.5% of the
  12.23 baseline) and footprint — exactly as expected, since FP8 is
  straightforward scaling. The gap is *operational*: LLM Compressor's
  `FP8_DYNAMIC` is data-free RTN (no calibration pass) and emits a
  `vllm serve`-ready checkpoint, vs ModelOpt's calibrated flow (~30 s for a
  128-sample amax pass) plus a separate export step.
* **NVFP4** roughly halves the footprint (0.452 GB vs 0.94 GB baseline). The
  perplexity rise (14.84) is the known cost of 4-bit on a *tiny* 0.5 B model —
  it shrinks sharply on larger models; re-run the harness with an 8 B model to
  see the gap close.
* **Portability mattered most here.** On a stock Windows + consumer-GPU box,
  LLM Compressor produced both FP8 and NVFP4 checkpoints with no extra build
  toolchain, while ModelOpt's microscale path required a C++ compiler it could
  not find. ModelOpt remains the stronger choice when you deploy through
  TensorRT-LLM on a fully provisioned NVIDIA stack.
<!-- RESULTS:END -->

### How to read the numbers

* **Perplexity** within noise of each other means both tools preserve accuracy
  equally for that scheme — expected for FP8, where both use straightforward
  per-tensor/channel scaling. Differences widen at 4-bit, where the *algorithm*
  (AWQ vs max-calibration) matters more than the tool.
* **Size** should be near-identical for the same numeric format; small deltas
  come from metadata and which modules each tool leaves in full precision.
* **Servability** is the practical differentiator: the LLM Compressor output is
  `vllm serve`-ready with no extra step.
