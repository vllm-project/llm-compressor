---
name: add-vllm-model
description: >
  Port a vLLM model definition into a transformers-compatible Custom Modeling
  Package under src/llmcompressor/modeling/ with DynamicCache,
  @use_experts_implementation, and a vLLM parity test.
args:
  arch:
    type: string
    description: Snake-case architecture name (e.g., "deepseek_v41")
    required: false
  model_id:
    type: string
    description: HuggingFace model ID or local checkpoint path for weight/config inspection
    required: false
---

# Add Custom Modeling Package from vLLM

Use a **vLLM Source Architecture** (`vllm/model_executor/models/<arch>.py`) to
create a **Custom Modeling Package** in `src/llmcompressor/modeling/<arch>/`
that works with `llmcompressor` (`oneshot`, `load_context`, KV cache quant, and
MoE linearization) and passes output parity against vLLM.

Read [shared_modeling.md](../shared_modeling.md) before writing any code. Every
contract in `shared_modeling.md` is mandatory.

## Step 1 — Gather Source Architecture & Checkpoint Metadata

1. Locate and read:
   - The vLLM model file (e.g., `vllm/model_executor/models/<arch>.py` and any
     config/helper modules it imports).
   - The checkpoint's `config.json` and `model.safetensors.index.json` (from
     `model_id` or HuggingFace Hub) to verify the exact `model_type`,
     `architectures`, config keys, and **on-disk tensor names and shapes**.
2. Inspect `load_weights(...)`, `packed_modules_mapping`, and
   `quant_config` handling in the vLLM file:
   - Map every vLLM fused/parallel layer back to the **actual checkpoint tensor
     names** (e.g., `QKVParallelLinear` -> `q_proj`, `k_proj`, `v_proj`;
     `MergedColumnParallelLinear` -> `gate_proj`, `up_proj`;
     `ColumnParallelLinear` / `RowParallelLinear` -> `torch.nn.Linear`).
   - Check bias flags (`bias=...`), norms (`RMSNorm`, offset norms, eps values),
     RoPE scaling / YaRN / mRoPE parameters, attention scaling, and sliding
     window or hybrid layer schedules.

## Step 2 — Create `src/llmcompressor/modeling/<arch>/`

Create `__init__.py`, `configuration_<arch>.py`, `modeling_<arch>.py`, and
`README.md` following [shared_modeling.md](../shared_modeling.md).

### vLLM-to-`transformers` Translation Rules

1. **Unwrap parallel & custom linear layers**:
   - Replace `ColumnParallelLinear`, `RowParallelLinear`, `ReplicatedLinear`,
     `QKVParallelLinear`, `MergedColumnParallelLinear` with standard
     `torch.nn.Linear` matching the checkpoint tensor names.
   - Replace `VocabParallelEmbedding` / `ParallelLMHead` with `nn.Embedding`
     and `nn.Linear` (respecting `tie_word_embeddings`).
2. **Unwrap vLLM attention & paged KV cache**:
   - Replace vLLM `Attention(...)` / PagedAttention / custom MLA kernels with
     standard PyTorch attention (`sdpa` / eager attention + RoPE) inside a class
     whose name contains `"Attention"` and exposes `k_proj`/`v_proj`/`qkv_proj`/
     `kv_b_proj`.
   - Accept `past_key_values: Cache | None = None`, instantiate
     `DynamicCache(config=self.config)` when `use_cache=True` and
     `past_key_values is None`, and update KV via `past_key_values.update(...)`.
3. **Unwrap vLLM `FusedMoE`**:
   - Replace `FusedMoE` with a router (`nn.Linear` or checkpoint-matched gate)
     plus an experts class decorated with `@use_experts_implementation`
     holding 3D `Parameter`s (`gate_up_proj` or `up_proj`, and `down_proj`).
   - Verify `MoEConfig.from_config(config)` succeeds on your `<Arch>Config`.
   - If the checkpoint on disk stores 2D per-expert weights, add 2D conversion
     mappings in `src/llmcompressor/modeling/moe/conversion_mappings.py` (or
     document `patch_moe_mappings`).
4. **Strip serving-only vLLM code**:
   - Remove `IntermediateTensors`, `PP` / `TP` `all_reduce` / `tensor_model_parallel`,
     `SupportsPP` / `SupportsQuant` mixins, and `load_weights` serving logic.
   - Preserve exact mathematical equivalence: residual placement, norm dtype
     casting (e.g., `.float()` inside RMSNorm), logit softcapping, scaling
     factors, and causal masking.
5. **Multimodal models** (if applicable):
   - Wrap vision/audio towers and text decoder inside
     `<Arch>ForConditionalGeneration` (subclassing `<Arch>PreTrainedModel` +
     `GenerationMixin`), implementing `get_input_embeddings`,
     `get_output_embeddings`, `get_decoder`, `set_decoder`, and `tie_weights`.

## Step 3 — Write Contract + vLLM Parity Tests

1. Copy [test_vllm_parity.py](templates/test_vllm_parity.py) to
   `tests/llmcompressor/modeling/test_<arch>.py`.
2. Fill in the tiny `<Arch>Config`, `<Arch>ForCausalLM` (or
   `<Arch>ForConditionalGeneration`), `architectures` field matching vLLM, and
   valid tokenizer source.
3. Ensure the parity test:
   - Tests **non-vLLM contracts unconditionally** (save/load round-trip via
     `load_context`, `is_cached_attention_module` + `DynamicCache` decode
     parity + `QuantizedKVCache` attachment, `linearize_moe` +
     `moe_calibration_context` + `repack_moe`, and `trace_subgraphs`).
   - Tests **vLLM parity** inside `@requires_gpu(1)` with
     `vllm = pytest.importorskip("vllm")` so `make test` CI skips cleanly when
     `vllm` is absent and validates prompt logprobs + greedy tokens when `vllm`
     is installed.
4. If validating locally against a real large checkpoint that exceeds VRAM, use
   `ExpertPruner` or `MagnitudeExpertPruner` as documented in
   [shared_modeling.md](../shared_modeling.md).

## Step 4 — Verify & Prepare PR

1. Run `make style` and `make quality` (see [style.md](../style.md)).
2. Run the new test file using `canhazgpu` if available (see [test.md](../test.md)):
   ```bash
   python3 -m pytest -v tests/llmcompressor/modeling/test_<arch>.py
   ```
3. Run the vLLM parity test inside an environment where `vllm` is installed and
   confirm prompt logprobs and greedy outputs match within tolerance.
4. Follow the PR checklist in [shared_modeling.md](../shared_modeling.md).