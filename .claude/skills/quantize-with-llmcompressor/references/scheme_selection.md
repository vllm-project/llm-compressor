# Scheme selection reference

This is the decision logic the skill uses to turn *hardware + model + goal* into
a concrete LLM Compressor scheme and algorithm. The authoritative scheme list
lives in `docs/guides/compression_schemes.md`
and in
[compressed-tensors `quant_scheme.py`](https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py).

## 1. What can my hardware *serve* efficiently?

Quantization (calibration) can run on almost any GPU, or CPU with offloading.
What follows gates the *deployment* GPU, because the numeric datapath must exist
in silicon to get a speedup. Run `scripts/detect_hardware.py` to classify the
local machine automatically.

| Architecture | Compute cap | Native datapaths | Best default scheme |
|---|---|---|---|
| Blackwell (RTX 50xx, B200, GB200) | 10.0 / 12.0 | FP4, FP8, INT8 | **NVFP4** |
| Hopper (H100/H200) | 9.0 | FP8, INT8 | **FP8_DYNAMIC** |
| Ada Lovelace (RTX 40xx, L40/L4) | 8.9 | FP8, INT8 | **FP8_DYNAMIC** |
| Ampere (A100, RTX 30xx) | 8.0 / 8.6 | INT8 | **W4A16** or INT8 W8A8 |
| Turing/Volta (RTX 20xx, T4, V100) | 7.0 / 7.5 | INT8 | **W4A16** |
| Older / CPU target | < 7.0 | weight-only | **W4A16** |

Weight-only schemes (W4A16 / W8A16) run on essentially any CUDA GPU through
Marlin kernels regardless of the table above — they just don't accelerate
activations.

## 2. What is my goal?

| Goal | Scheme | Algorithm | Needs calibration data? |
|---|---|---|---|
| Max throughput on Blackwell | NVFP4 (W4A4) | RTN | Yes (activation global scale) |
| Best perf/accuracy on Ada/Hopper | FP8_DYNAMIC | RTN | **No** (data-free) |
| Large / MoE model, FP8 | FP8_BLOCK | RTN | No |
| Max compression, latency-bound | W4A16 | GPTQ or AWQ | Yes |
| Safest accuracy at 4-bit weights | NVFP4A16 / W4A16 | AWQ | Yes |
| High-QPS INT8 serving | W8A8 (INT8) | GPTQ + SmoothQuant | Yes |
| Brand-new model / no HF def / huge | FP8_BLOCK | model_free_ptq | No |

Rules of thumb:

* **Prefer data-free first.** FP8_DYNAMIC needs no dataset and is hard to beat
  on Ada/Hopper. Only reach for GPTQ/AWQ when you need 4-bit weights.
* **RTN vs GPTQ/AWQ.** RTN is instant and data-free; GPTQ/AWQ cost calibration
  time but recover accuracy at 4-bit. AWQ is usually the strongest for W4A16.
* **NVFP4 vs NVFP4A16.** `NVFP4` quantizes activations too (W4A4, fastest, needs
  calibration). `NVFP4A16` keeps activations in FP16 (data-free, more accurate,
  slightly slower) — good default when accuracy matters.

## 3. Algorithm → modifier mapping

| Algorithm | Modifier | Import |
|---|---|---|
| RTN (round-to-nearest) | `QuantizationModifier` | `llmcompressor.modifiers.quantization` |
| GPTQ | `GPTQModifier` | `llmcompressor.modifiers.gptq` |
| AWQ | `AWQModifier` (+ `QuantizationModifier`) | `llmcompressor.modifiers.transform.awq` |
| SmoothQuant (pairs with INT8) | `SmoothQuantModifier` | `llmcompressor.modifiers.transform.smoothquant` |

> AWQ and SmoothQuant moved to `llmcompressor.modifiers.transform.*`; the old
> `llmcompressor.modifiers.{awq,smoothquant}` paths are deprecated shims. AWQ is a
> transform that does not quantize on its own — pair it with a `QuantizationModifier`
> in a list recipe: `recipe = [AWQModifier(duo_scaling="both"),
> QuantizationModifier(scheme="W4A16_ASYM", targets=["Linear"], ignore=["lm_head"])]`
> (the canonical Llama AWQ example uses `W4A16_ASYM`; Qwen examples use `W4A16`).
> `scripts/verify_apis.py` checks these imports against your installed packages.

## 4. Model-specific gotchas

* **MoE models**: calibration-friendly expert handling is now applied
  **automatically by the pipeline** — you do *not* import or instantiate any
  `Calibration*MoE` class. Just build a normal `GPTQModifier` /
  `QuantizationModifier` / AWQ recipe and add the router/gate to `ignore`
  (e.g. `ignore=["lm_head", "re:.*mlp.gate$"]`). See
  `examples/quantizing_moe`.
* **Vision models** load via `AutoModelForImageTextToText` (or a model-specific
  class like `Gemma3ForConditionalGeneration`) and should ignore the vision
  tower and projector, e.g. `["lm_head", "re:.*vision_tower.*",
  "re:.*multi_modal_projector.*"]`. Data-free schemes (FP8_DYNAMIC, NVFP4A16,
  W*A16) need no dataset; **calibrated** vision (GPTQ/AWQ/NVFP4) requires a
  model-specific `data_collator` (or the `processor=`/`dataset=` framework path)
  — follow `examples/multimodal_vision`
  directly rather than generic text calibration.
* **Audio models** are model-specific: whisper uses
  `WhisperForConditionalGeneration` + `WhisperProcessor` + a `data_collator`.
  Follow `examples/multimodal_audio` — generic text
  calibration will not calibrate the audio pathway.
* **Always ignore `lm_head`** (and `re:.*mlp.gate$` for MoE routers).
* **Very large models**: use `model_free_ptq` (data-free schemes) or sequential
  onloading to avoid OOM. See `examples/model_free_ptq` and
  `examples/big_models_with_sequential_onloading`.

## 5. VRAM sizing for the calibration run

Naive `oneshot` holds the model in BF16 (~2 bytes/param) plus activation
overhead, so a single GPU handles roughly `VRAM_GB / 2.6` billion parameters
directly. Sequential onloading and `model_free_ptq` stream layers and push this
far higher. `detect_hardware.py` prints a per-GPU estimate.

For a concrete per-model answer, run `scripts/estimate_fit.py`, which reads the
exact parameter count from the model's Hugging Face safetensors metadata (no
download) and checks the quantized footprint against VRAM for both serving and
calibration — on the local GPU, a named `--target` GPU, or an explicit
`--vram-gb`. This is the LLM Compressor / compressed-tensors analogue of
[Hugging Face's per-hardware model-fit panel](https://huggingface.co/docs/hub/main/hardware),
which only covers GGUF/MLX files. Known target GPUs:
`detect_hardware.py --list-targets`.
