---
name: add-pytorch-model
description: >
  Port a standalone torch.nn.Module definition into a transformers-compatible
  Custom Modeling Package under src/llmcompressor/modeling/ with DynamicCache,
  @use_experts_implementation, and a PyTorch parity test.
args:
  arch:
    type: string
    description: Snake-case architecture name (e.g., "deepseek_v32")
    required: false
  model_id:
    type: string
    description: HuggingFace model ID or local checkpoint path for weight/config inspection
    required: false
---

# Add Custom Modeling Package from `torch.nn.Module`

Use a standalone PyTorch **Source Architecture** (for example, an author's
`inference/model.py` using raw `torch.nn.Module` and dataclasses) to create a
**Custom Modeling Package** in `src/llmcompressor/modeling/<arch>/` that works
with `llmcompressor` and matches the original PyTorch module's outputs.

Read [shared_modeling.md](../shared_modeling.md) before writing any code. Every
contract in `shared_modeling.md` is mandatory.

## Step 1 — Gather Source Architecture & Checkpoint Metadata

1. Locate and read:
   - The standalone PyTorch `nn.Module` definition and its config dataclass
     (e.g., `ModelArgs`) or conversion script (e.g., `convert.py`).
   - The checkpoint's `config.json` and `model.safetensors.index.json` (or
     `.pt`/`.safetensors` keys) to verify exact `model_type`, config keys, and
     **on-disk tensor names and shapes**.
2. Identify every adaptation needed:
   - Tensor/pipeline parallel layers or `torch.distributed` calls (`all_reduce`,
     `ColumnParallelLinear`, `RowParallelLinear`, `ParallelEmbedding`).
   - Custom FP8 / quantized linear wrappers or Triton/TileLang inference kernels
     that need standard `bfloat16`/`float16` PyTorch equivalents.
   - Manual static KV buffers (`self.kv_cache`, `start_pos`) that must be
     replaced with `DynamicCache`.
   - Custom MoE loops or `nn.ModuleList` experts that must be replaced with
     `@use_experts_implementation` + 3D `Parameter`s.

## Step 2 — Create `src/llmcompressor/modeling/<arch>/`

Create `__init__.py`, `configuration_<arch>.py`, `modeling_<arch>.py`, and
`README.md` following [shared_modeling.md](../shared_modeling.md).

### PyTorch-to-`transformers` Translation Rules

1. **Convert dataclass config to `PreTrainedConfig`**:
   - Convert `ModelArgs` / `@dataclass` configs to `<Arch>Config(PretrainedConfig)`
     with `model_type = "<arch>"` and standard HF attribute names (`hidden_size`,
     `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
     `vocab_size`, MoE fields).
   - Merge any split inference config JSON keys cleanly or document defaults in
     `configuration_<arch>.py`.
2. **Standardize layers & remove distributed assumptions**:
   - Replace parallel linear/embedding wrappers with `torch.nn.Linear` and
     `torch.nn.Embedding` (respecting `bias=False` defaults from custom Linear
     classes!).
   - Remove `world_size > 1` / rank slicing so each module holds full unsharded
     weights compatible with `oneshot` DDP and sequential offloading.
   - Replace hardcoded inference-only FP8/Triton kernels with clean PyTorch
     primitives (keep custom helper kernels in a separate `kernel.py` if needed,
     matching `bfloat16` reference numerics).
3. **Integrate `DynamicCache` & `@use_experts_implementation`**:
   - Replace raw `self.kv_cache` buffers with `DynamicCache` +
     `past_key_values.update(...)` inside a class named `*Attention` exposing
     `k_proj`/`v_proj`/`qkv_proj`/`kv_b_proj`.
   - Replace custom MoE expert blocks with `@use_experts_implementation` and 3D
     `gate_up_proj` / `up_proj` + `down_proj` `Parameter`s.
4. **Wrap with `<Arch>PreTrainedModel` & `<Arch>ForCausalLM` (or
   `<Arch>ForConditionalGeneration`)**:
   - Inherit `(PreTrainedModel, GenerationMixin)` (or `<Arch>PreTrainedModel` +
     `GenerationMixin`), set `base_model_prefix`, `_no_split_modules`,
     `_skip_keys_device_placement = ["past_key_values"]`, implement embedding
     getters/setters, and return `CausalLMOutputWithPast` (with `logits`,
     `past_key_values`, and optional `loss`).
   - Ensure parameter names match the target checkpoint keys (or provide explicit
     conversion/renaming notes in `README.md` if converting from raw author
     `.pt` weights).

## Step 3 — Write Contract + PyTorch Parity Tests

1. Copy [test_pytorch_parity.py](templates/test_pytorch_parity.py) to
   `tests/llmcompressor/modeling/test_<arch>.py`.
2. Wire the tiny `<Arch>Config`, `<Arch>ForCausalLM`, and reference
   `torch.nn.Module` weight mapping into the parity test.
3. Ensure the test validates:
   - Unconditional LLM Compressor contracts (round-trip via `load_context`,
     `DynamicCache` + `QuantizedKVCache`, `@use_experts_implementation`
     linearization + `moe_calibration_context` + `repack_moe`, and
     `trace_subgraphs`).
   - Exact in-process PyTorch parity (`torch.testing.assert_close` on prefill
     logits and multi-step cached decode) against the original `torch.nn.Module`
     on identical weights and inputs.
4. If testing locally against a real large checkpoint that exceeds VRAM, use
   `ExpertPruner` or `MagnitudeExpertPruner` as documented in
   [shared_modeling.md](../shared_modeling.md).

## Step 4 — Verify & Prepare PR

1. Run `make style` and `make quality` (see [style.md](../style.md)).
2. Run the test suite (using `canhazgpu` if available; see [test.md](../test.md)):
   ```bash
   python3 -m pytest -v tests/llmcompressor/modeling/test_<arch>.py
   ```
3. Follow the PR checklist in [shared_modeling.md](../shared_modeling.md).