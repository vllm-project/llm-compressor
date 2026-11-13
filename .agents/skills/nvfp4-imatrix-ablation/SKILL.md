---
name: nvfp4-imatrix-ablation
description: >-
  Reproduce NVFP4 imatrix quantization and benchmark evaluations for the
  basic, groupmax, and inverse-groupmax objectives using this repository's
  model-specific recipes and the eval-agent skill.
---

# NVFP4 iMatrix ablation

Use this skill when reproducing the NVFP4 iMatrix objective experiments in
`examples/imatrix`. The quantization scripts are in this repository. For
benchmark setup, serving, smoke tests, and monitoring, follow the external
`eval-agent` skill; do not edit its task registry or change task definitions.

## Objectives

- `basic`: `importance * abs(Q(w) - w) ** power`
- `groupmax`: the basic weight multiplied by `sqrt(w ** 2 + groupmax ** 2)`
- `groupmax_div`: the basic weight divided by `sqrt(w ** 2 + groupmax ** 2)`

`groupmax` is the original absolute maximum of the element's quantization group.
The groupmax objectives require grouped weight quantization.

## Model profiles

| Script model key | Hugging Face model | Available profiles |
| --- | --- | --- |
| `qwen38_27b` | `Qwen/Qwen3.8-27B` | `all` |
| `qwen3_30b_a3b` | `Qwen/Qwen3-30B-A3B` | `all`, `attention_only` |
| `nemotron35_lightning_30b_a3b` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | `all`, `attention_only` |
| `gemma4_26b_a4b_it` | `google/gemma-4-26B-A4B-it` | `all`, `attention_only` |
| `qwen25_7b_instruct` | `Qwen/Qwen2.5-7B-Instruct` | `all` |

The published full grid uses powers `2, 2.2, 2.4, 2.6, 2.8, 3`, all three
objectives, and the eight model/profile combinations above (144 conditions).
Both one-shot and GPTQ quantization are supported by the corresponding
`quantize_nvfp4_*_imatrix_variant.py` scripts.

## Quantize one condition

First follow the repository's NVFP4 and model-specific quantization setup.
Use an environment with this checkout installed, compatible Transformers and
PyTorch versions, and the model's required dependencies. The recipe uses 512
`perfectblend` training samples with a 4096-token maximum sequence length.
For MoE checkpoints it calibrates all experts where supported. Check the
checkpoint's model card for access approval and revision details.

Choose the quantization method and run one of:

```bash
python examples/imatrix/quantize_nvfp4_imatrix_variant.py \
  --model gemma4_26b_a4b_it --power 2.4 --objective groupmax_div \
  --module-profile attention_only --variant-dir ./runs/gemma4-p2.4-div
```

```bash
python examples/imatrix/quantize_nvfp4_gptq_imatrix_variant.py \
  --model gemma4_26b_a4b_it --power 2.4 --objective groupmax_div \
  --module-profile attention_only --variant-dir ./runs/gemma4-p2.4-div-gptq
```

The result is written to `<variant-dir>/checkpoint`. Preserve that checkpoint
and its `imatrix_ablation_config.json`; the metadata is needed to validate the
module profile and reproduce the run. Use `groupmax_div` as the objective name;
`groupmaxinv` is accepted as a legacy alias.

## Evaluate a checkpoint

Read and apply the external `eval-agent` skill first. It specifies the required
evaluation environments, vLLM setup, GPU reservation, smoke test, KV-cache
calibration, monitoring, and retry procedure. Use its task registry unchanged.
The reproduction wrapper uses the installed eval-agent Python API so it can
run the research seed schedule without editing registry settings. Run it from
the eval-agent base environment (for example, `.venv/bin/python`) so
`eval_agent` can be imported.

Run a smoke evaluation before full evaluation. Select the category from the
model family: Qwen3 and Nemotron models use `reasoning`; Gemma 4 and Qwen2.5
use `instruct`. Use the vLLM server recipe/chat template recommended by the
model card or vLLM model recipe. The server command must serve the checkpoint
and include the appropriate parser/template flags. Wrap it with the local GPU
reservation command when the environment provides one, for example
`chg run --gpus 1 -- vllm serve ...`.

Use the eval-agent skill's `max_model_len` formula, capped by the model's
supported context length (the experiment used 20,480 for instruct models and
69,632 for reasoning models). Model-specific serving flags used in this grid:

- Qwen3-30B-A3B and Qwen3.8-27B: `--reasoning-parser qwen3`.
- Nemotron 3.5 Lightning: `--reasoning-parser nemotron_v3`, the model's
  `chat_template.jinja`, and chat-template kwargs
  `{"enable_thinking":true,"force_nonempty_content":true}`.
- Gemma 4: the Gemma4 tool chat template and chat-template kwargs
  `{"enable_thinking":false}`. Gemma 4 attention-only also needs
  `--enforce-eager` with the vLLM version used by the research run.
- Qwen2.5-7B-Instruct: use the model's published chat template and generation
  configuration.

```bash
python examples/imatrix/evaluate_nvfp4_imatrix_checkpoint.py \
  --model gemma4_26b_a4b_it \
  --checkpoint ./runs/gemma4-p2.4-div/checkpoint \
  --run-dir ./runs/gemma4-p2.4-div/eval-smoke \
  --phase smoke --module-profile attention_only \
  --server-cmd "chg run --gpus 1 -- vllm serve ./runs/gemma4-p2.4-div/checkpoint --tensor-parallel-size 1 --dtype auto --max-model-len 20480 --gpu-memory-utilization 0.9 --max-num-seqs 32 --port 8000 --trust-remote-code --enforce-eager --chat-template /path/to/gemma4_chat_template.jinja --default-chat-template-kwargs '{\"enable_thinking\":false}'" \
  --max-length 20480 --num-concurrent 32 --port 8000 \
  --lm-eval-venv .venvs/lm-eval --lighteval-venv .venvs/lighteval
```

For Gemma 4 attention-only checkpoints, the wrapper rewrites the checkpoint's
vLLM runtime targets to account for vLLM's module-path mapping and the five
full-attention layers with shared K/V weights. The wrapper does not change the
saved packed weights. Use the `--enforce-eager` server flag for this profile to
avoid the vLLM compile stride issue observed in the research run.

After the smoke passes and concurrency is calibrated, run full evaluation
once for each seed. The first run includes MMLU; later runs omit MMLU while
retaining all other tasks. Keep a distinct run directory for each seed:

```bash
SERVER_CMD="chg run --gpus 1 -- vllm serve ./runs/gemma4-p2.4-div/checkpoint --tensor-parallel-size 1 --dtype auto --max-model-len 20480 --gpu-memory-utilization 0.9 --max-num-seqs 32 --port 8000 --trust-remote-code --enforce-eager --chat-template /path/to/gemma4_chat_template.jinja --default-chat-template-kwargs '{\"enable_thinking\":false}'"

for seed in 1234 2345 3456; do
  python examples/imatrix/evaluate_nvfp4_imatrix_checkpoint.py \
    --model gemma4_26b_a4b_it \
    --checkpoint ./runs/gemma4-p2.4-div/checkpoint \
    --run-dir "./runs/gemma4-p2.4-div/eval-seed-${seed}" \
    --phase full --seed "$seed" --module-profile attention_only \
    --server-cmd "$SERVER_CMD" \
    --max-length 20480 --num-concurrent 32 --port 8000 \
    --lm-eval-venv .venvs/lm-eval --lighteval-venv .venvs/lighteval
done
```

The eval-agent registry determines prompts, few-shot counts, generation limits,
metrics, and task repetitions. The wrapper selects MMLU only for seed 1234 and
uses the registry definitions for all task settings. Inspect each run's
`summary_data.json`, manifest, task logs, and `events.jsonl`; do not treat a
failed or incomplete task as a score.

## Model-specific checks

- The quantizer checks that routers and `lm_head` remain unquantized.
- Nemotron uses `NVFP4A16`; its attention/mixer, convolution, embeddings,
  latent projections, router/gate, MTP layers, final norm, and `lm_head` follow
  the exclusions in the checked-in recipe.
- Attention-only profiles validate the packed-module allowlist. They are
  intended for Qwen3-30B-A3B, Nemotron 3.5 Lightning, and Gemma 4 only.
- Keep the quantization method, objective, power, calibration data, profile,
  package versions, vLLM flags, and eval-agent run artifacts with reported
  results.
