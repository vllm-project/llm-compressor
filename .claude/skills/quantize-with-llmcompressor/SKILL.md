---
name: quantize-with-llmcompressor
description: >-
  Detect what quantization a machine's GPU can efficiently run, recommend the
  best LLM Compressor scheme for a given model and goal, and generate (and
  optionally run) a canonical examples/-style quantization script that produces a
  vLLM-ready compressed-tensors checkpoint. Use whenever a user wants to quantize
  / compress an LLM or VLM, asks which quantization format their hardware supports
  (FP8, NVFP4, W4A16, INT8, MXFP4), wants to shrink a model for vLLM, wants to
  write or update a quantization example, or asks how LLM Compressor compares to
  NVIDIA TensorRT Model Optimizer.
version: 0.2.0
license: Apache-2.0
metadata:
  data-classification: public
  author: LLM Compressor contributors
  tags: [quantization, llmcompressor, vllm, fp8, nvfp4, w4a16, hardware]
  domain: model-optimization
---

# Quantize with LLM Compressor

Turn "I want to make this model smaller/faster" into a correct, hardware-matched
quantized checkpoint that runs in vLLM — without the user having to know which of
a dozen schemes and algorithms applies to their GPU.

## When to use this skill

Trigger when the user wants to quantize, compress, or shrink an LLM/VLM; asks
which formats (FP8, NVFP4, W4A16, INT8, MXFP4/MXFP8) their hardware supports;
wants a vLLM-ready checkpoint; or asks how LLM Compressor stacks up against
NVIDIA Model Optimizer.

## Workflow

Follow these steps in order. Do not skip step 1 — recommending a scheme without
knowing the hardware is the most common mistake.

### Step 1 — Detect the hardware

```bash
python .claude/skills/quantize-with-llmcompressor/scripts/detect_hardware.py
```

This classifies every local GPU by compute capability into an architecture
(Blackwell / Hopper / Ada / Ampere / Turing / older), reports VRAM, and prints
both the schemes it can **efficiently serve** and a single **recommended**
default. Add `--json` for machine-readable output.

Quantizing for a GPU you are **not** on (e.g. on a laptop, deploying to an
H100)? Use a named target instead of local detection:

```bash
python .claude/skills/quantize-with-llmcompressor/scripts/detect_hardware.py --target h100
python .claude/skills/quantize-with-llmcompressor/scripts/detect_hardware.py --list-targets
```

The target specs are a small curated table (architecture / VRAM / native
datapaths). For the broader community hardware catalog and a user's own declared
setup, see Hugging Face's hardware pages
(<https://huggingface.co/docs/hub/main/hardware>,
<https://huggingface.co/hardware>).

Key distinction to communicate to the user:

* Calibration/quantization can run on almost any GPU (or CPU with offloading).
* The *serving* speedup is gated by silicon: **FP4 needs Blackwell, FP8 needs
  Ada/Hopper+, INT8 needs Turing+**. Weight-only W4A16/W8A16 runs anywhere.
* It is fine to quantize **for a different deployment GPU** than the local one —
  pick the scheme for the *serving* target.

**Never fabricate device specs.** If detection fails, say so and fall back to the
safe, universally supported default (W4A16) rather than guessing.

### Step 2 — Choose the scheme

Map hardware + model + goal to a scheme using
[references/scheme_selection.md](references/scheme_selection.md). Short version:

| Serving GPU | Default | If you need more compression |
|---|---|---|
| Blackwell | `NVFP4` (or `NVFP4A16` for accuracy) | already 4-bit |
| Ada / Hopper | `FP8_DYNAMIC` (data-free) | `W4A16` (AWQ) |
| Ampere / older | `W4A16` (AWQ) | `W8A16` for safety |

Prefer **data-free** schemes first (`FP8_DYNAMIC`, `W*A16` RTN, `NVFP4A16`): no
dataset, instant. Reach for GPTQ/AWQ only when you need accurate 4-bit weights.
Confirm the chosen scheme + goal with the user before a long calibration run.

### Step 3 — Check it will fit (before downloading anything)

Estimate the quantized footprint and whether it fits VRAM for **both** serving
and the calibration run — using the model's real parameter count read from the
Hugging Face Hub (no download):

```bash
python .claude/skills/quantize-with-llmcompressor/scripts/estimate_fit.py \
    --model meta-llama/Llama-3.1-8B-Instruct --scheme NVFP4          # local GPU
python .claude/skills/quantize-with-llmcompressor/scripts/estimate_fit.py \
    --model Qwen/Qwen2.5-72B-Instruct --scheme W4A16 --target a100   # off-machine
```

If naive `oneshot` won't fit, it tells you to switch to sequential onloading or
`model_free_ptq`. This catches OOM before a long download/run rather than after.

### Step 4 — Generate the example script (and optionally run it)

The skill does **not** wrap llmcompressor in a bespoke engine. Instead it
*generates a flat example script in the exact style of the repo's `examples/`*,
copied from the current canonical templates, so the output uses only verified,
non-deprecated APIs and is itself a mergeable "day-1" example. This is how the
skill helps you author or update an example instead of hand-writing one.

```bash
# Data-free FP8 (Ada/Hopper) — emits a canonical example, no dataset:
python .claude/skills/quantize-with-llmcompressor/scripts/generate_example.py \
    --model meta-llama/Llama-3.1-8B-Instruct --scheme FP8_DYNAMIC -o fp8_example.py

# NVFP4 on Blackwell (calibrated) — write AND run it on your GPU:
python .claude/skills/quantize-with-llmcompressor/scripts/generate_example.py \
    --model meta-llama/Llama-3.1-8B-Instruct --scheme NVFP4 -o nvfp4_example.py --run

# Max compression W4A16 with AWQ (two-modifier transform recipe):
python .claude/skills/quantize-with-llmcompressor/scripts/generate_example.py \
    --model meta-llama/Llama-3.1-8B-Instruct --scheme W4A16_ASYM --algorithm awq \
    -o awq_example.py

# Vision-language model, data-free:
python .claude/skills/quantize-with-llmcompressor/scripts/generate_example.py \
    --model google/gemma-3-4b-it --scheme NVFP4A16 --modality vision \
    -o vision_example.py
```

Flags: `--algorithm {rtn,gptq,awq}`, `--modality {text,vision}`, `--extra-ignore`
(e.g. MoE gate `re:.*mlp.gate$`), `--num-samples`, `--run`, `--list-references`.

**New model or new format? Don't trust the template — adapt a real example.**
The generator emits a *known-good pattern with generic ignores* (`lm_head`); it
does **not** infer model-specific needs (special ignore patterns, a non-`CausalLM`
class, MoE expert handling, format-specific steps like NVFP4 reindexing). To stay
correct against the repo *as it is now*, the skill grounds itself in live code:

* it **validates the scheme against the installed compressed-tensors registry**
  and refuses unknown/renamed schemes (run `scripts/verify_apis.py` to list them);
* it **surfaces the nearest current `examples/` scripts** for your model + scheme
  — every generate run prints them, or use `--list-references` to just see them.

So for a genuinely new model family or quantization format: run
`--list-references`, **read the closest current example**, and adapt it (model
class, ignores, collator, MoE) rather than shipping the raw template. Calibrated
vision (GPTQ/AWQ/NVFP4) and audio (whisper) in particular need a model-specific
data collator — adapt `examples/multimodal_vision` / `examples/multimodal_audio`.
For huge models, use `model_free_ptq` / sequential onloading (see
[references/scheme_selection.md](references/scheme_selection.md)).

### Step 5 — Verify and deploy

The generated script runs a sample generation and saves a `compressed-tensors`
checkpoint. Serve it directly:

```bash
vllm serve <model>-<SCHEME>
```

To quantize **many** models unattended, drive this skill from the `/loop` skill
(one model per iteration) and stop when the batch is done.

### Step 6 — (Optional) Compare with NVIDIA Model Optimizer

This is an **optional dev/eval tool**, deliberately outside the core skill: it
pulls in `nvidia-modelopt` (not an upstream dependency) and is best run per-tool
in its own venv.

```bash
python .claude/skills/quantize-with-llmcompressor/scripts/benchmark_compare.py \
    --model meta-llama/Llama-3.2-1B-Instruct --scheme FP8_DYNAMIC \
    --tools llmcompressor
```

It reports WikiText-2 perplexity, checkpoint size, and quantization time. The
built-in perplexity is a quick proxy — use `lm-eval` for citable accuracy. See
[references/modelopt_comparison.md](references/modelopt_comparison.md) for the
capability comparison and measured numbers — the headline is that LLM Compressor
output is `vllm serve`-ready with no engine build.

## Trust & verification

This skill is designed to be easy for a maintainer — and a user — to trust:

* **It mirrors verified repo code.** `generate_example.py`'s templates are copied
  from the current canonical `examples/` (FP8, GPTQ, AWQ, NVFP4, INT8, vision) —
  it does not reimplement library internals.
* **One-command API check.** Run
  `python .claude/skills/quantize-with-llmcompressor/scripts/verify_apis.py` to confirm
  every import and scheme string the skill uses is real in the *installed*
  llmcompressor / compressed-tensors (it imports nothing remote and loads no
  model). Use this whenever upgrading those packages.

### Safety properties

* **No telemetry, no phone-home.** Nothing reports usage or sends data anywhere.
* **Network is limited to Hugging Face you already use.** The only network
  access is the same HF Hub reads any llmcompressor example performs — model/
  dataset downloads and public metadata — and only for runs you start.
* **Read-only inspection.** `detect_hardware.py`, `estimate_fit.py`, and
  `verify_apis.py` only introspect the GPU / installed packages / public HF
  metadata; they load no model and modify nothing.
* **Writes only where you point it.** `generate_example.py` writes the single
  `-o` file you name; `--run` executes *that* local file with the current Python
  (`subprocess` list form, no shell, no `shell=True`) — there is no hidden or
  remote code execution.
* **No credentials handled.** The scripts never read or transmit tokens/secrets.
* **Output is transparent.** The generated script is a plain, reviewable
  `examples/`-style file you can read before running.

## Bundled resources

* `scripts/detect_hardware.py` — GPU (local or `--target`) → architecture →
  supported/recommended schemes.
* `scripts/estimate_fit.py` — quantized footprint + VRAM fit check from the
  model's HF parameter count (no download).
* `scripts/generate_example.py` — emit a canonical-style example script
  (optionally `--run` it) using current, verified APIs.
* `scripts/verify_apis.py` — prove the skill references only real APIs/schemes.
* `scripts/benchmark_compare.py` — optional dev tool: LLM Compressor vs NVIDIA
  Model Optimizer.
* `references/scheme_selection.md` — full hardware/goal → scheme decision logic.
* `references/modelopt_comparison.md` — capability + empirical comparison.

## Platform notes

* **FP4 needs Triton.** The FP4 cast is `torch.compile`-decorated; on platforms
  without Triton (e.g. Windows) it raises `TritonMissing`. `generate_example.py`
  (`--run`) and `benchmark_compare.py` auto-detect this and fall back to eager
  (`TORCHDYNAMO_DISABLE=1`), so NVFP4/MXFP4 still run — just without compiled
  kernels.
* **Blackwell needs the cu128 build.** Install torch from the cu128 index
  (`--index-url https://download.pytorch.org/whl/cu128`); installing other
  packages afterward can silently pull a CPU/older-CUDA torch and break the GPU.
* **The ModelOpt comparison needs its own venv.** `nvidia-modelopt` pins an
  older `transformers` than `llmcompressor` requires, so run each tool in a
  separate environment (see references/modelopt_comparison.md).

## Hard constraints

* Never invent GPU model names, VRAM, or compute capabilities — read them from
  `detect_hardware.py` / `nvidia-smi`.
* Never claim a scheme is hardware-accelerated on a GPU that lacks the datapath
  (e.g. NVFP4 on non-Blackwell, FP8 on Ampere); say it will run weight-only or
  emulated instead.
* Always keep `lm_head` (and MoE routers) in full precision unless the user
  explicitly asks otherwise.
* Confirm before kicking off a long GPTQ/AWQ calibration run.
