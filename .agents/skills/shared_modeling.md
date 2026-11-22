# Shared Custom Modeling Guide

This guide defines the shared architecture, contracts, testing, and quality
standards for adding a **Custom Modeling Package** under
`src/llmcompressor/modeling/<arch>/`. Both the `add-vllm-model` and
`add-pytorch-model` skills must follow every contract below.

## Terminology

- **Source Architecture**: The upstream `vllm` model class or standalone
  `torch.nn.Module` implementation being ported.
- **Custom Modeling Package**: A self-contained `PreTrainedConfig` +
  `PreTrainedModel` package placed in `src/llmcompressor/modeling/<arch>/`.
- **MoE Calibration Definition**: A `LinearExperts2D` subclass in
  `src/llmcompressor/modeling/moe/` (only used when `@use_experts_implementation`
  cannot apply).
- **Tiny Verification Model**: A small synthetic-config model (few layers,
  small hidden size) used for fast, self-contained unit and CI tests.
- **Pruned Checkpoint**: A real checkpoint shrunk on disk via `ExpertPruner` or
  `MagnitudeExpertPruner` for local parity checks under limited VRAM.

---

## Step 1 — Package Layout & Registration

Create a directory `src/llmcompressor/modeling/<arch>/` using Hugging Face file
naming:

```text
src/llmcompressor/modeling/<arch>/
├── __init__.py
├── configuration_<arch>.py
├── modeling_<arch>.py
└── README.md
```

Rules:
1. **Export public classes in `__init__.py`**: Export `<Arch>Config` and the
   top-level model class (`<Arch>ForCausalLM` or
   `<Arch>ForConditionalGeneration`).
2. **Do NOT mutate global registries on `import llmcompressor`**: Do not import
   `<arch>` inside `src/llmcompressor/modeling/__init__.py` and do not call
   `AutoConfig.register` / `AutoModelForCausalLM.register` globally at import
   time. Users load the model explicitly with:
   ```python
   with load_context(<Arch>ForCausalLM):
       model = <Arch>ForCausalLM.from_pretrained(MODEL_ID)
   ```
3. **Checkpoint weight names are the ground truth**: `state_dict` parameter and
   buffer names in `modeling_<arch>.py` MUST match the actual Hugging Face
   checkpoint `.safetensors` keys on disk (not `vllm`'s internal fused names
   such as `qkv_proj` unless the checkpoint itself stores fused weights).
4. **Document provenance in `README.md`**: Record the upstream source commit or
   repo, any adaptations made (kernel fallbacks, TP removal, cache changes), and
   any special loading notes.
5. **Code quality**: All new files must pass `make style` and `make quality`
   (see [style.md](style.md)). Do not add new exclusions to `pyproject.toml`.

---

## Step 2 — Mandatory `transformers` & LLM Compressor Contracts

### 1. `PreTrainedConfig` (`configuration_<arch>.py`)
- Subclass `transformers.PreTrainedConfig` (or `PretrainedConfig`, its alias).
- Set `model_type = "<arch>"` matching the checkpoint's `config.json`.
- Accept and preserve all hyperparameters from `config.json` in `__init__`,
  passing remaining `**kwargs` (including `tie_word_embeddings`, token IDs,
  `dtype` / `torch_dtype`, etc.) to `super().__init__(**kwargs)`.
- For multimodal models, nest `text_config` and `vision_config` (or
  `audio_config`) sub-configs cleanly and support dict-to-config conversion in
  `__init__`.
- Ensure standard attributes expected by `compressed_tensors` and
  `llmcompressor.modeling.moe.helpers.MoEConfig.from_config` are exposed on the
  decoder/text config:
  - Attention: `num_attention_heads`, `num_key_value_heads`, `head_dim` (or
    `hidden_size // num_attention_heads`).
  - MoE (if applicable): `num_experts` (or `num_local_experts` /
    `n_routed_experts`), `num_experts_per_tok` (or `top_k_experts`),
    `hidden_size`, `moe_intermediate_size` (or `intermediate_size`),
    `hidden_act`, and `dtype`.

### 2. `DynamicCache` Integration (KV Cache Quantization Support)
`compressed_tensors.modeling.kvcache.QuantizedKVCache` attaches to attention
modules only when `is_cached_attention_module(module)` returns `True`:
1. **Attention class name**: Must contain `"attention"` case-insensitively
   (e.g., `<Arch>Attention`, NOT `MLA` or `SelfAttn` without `"Attention"`).
2. **Projection attribute name**: Must expose at least one attribute in
   `("k_proj", "v_proj", "qkv_proj", "kv_b_proj")` (`_KV_CACHE_PROJECTION_NAMES`).
3. **`forward` signature and cache update**:
   - Accept `past_key_values: Cache | None = None` in `Attention.forward` and
     the decoder `Model.forward`.
   - When `use_cache=True` and `past_key_values is None` in `Model.forward`,
     instantiate `DynamicCache(config=self.config)` from
     `transformers.cache_utils`.
   - Inside `Attention.forward`, store `self.layer_idx = layer_idx` and update
     the cache via:
     ```python
     if past_key_values is not None:
         cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
         key_states, value_states = past_key_values.update(
             key_states, value_states, self.layer_idx, cache_kwargs
         )
     ```
   - Never use raw static tensor buffers (such as `self.register_buffer("kv_cache", ...)`
     with `start_pos`) for standard attention.
   - Keep `PreTrainedModel._supports_default_dynamic_cache()` as `True` (do not
     override it to `False` or set a non-`DynamicCache` `_cache_class` unless
     strictly required by hybrid recurrent/SSM state).
   - For hybrid architectures (e.g., linear/delta attention + full attention),
     keep `DynamicCache` semantics on standard attention layers and subclass
     `DynamicCache` only for the extra recurrent states so `QuantizedKVCache`
     still wraps attention `.update(...)`.

### 3. `@use_experts_implementation` Integration (MoE Linearization Support)
LLM Compressor's `load_context()` uses `FusedExpertsProtocol` to automatically
unfuse 3D expert tensors into 2D `LinearExperts2D` modules so every expert is
calibrated:
1. **Decorate the experts class directly** with `@use_experts_implementation`
   from `transformers.integrations.moe`:
   ```python
   from transformers.integrations.moe import use_experts_implementation

   @use_experts_implementation
   class <Arch>Experts(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.num_experts = config.num_experts
           self.hidden_dim = config.hidden_size
           self.intermediate_dim = config.moe_intermediate_size
           self.gate_up_proj = nn.Parameter(
               torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
           )
           self.down_proj = nn.Parameter(
               torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
           )
           self.act_fn = ACT2FN[config.hidden_act]

       def forward(
           self,
           hidden_states: torch.Tensor,
           top_k_index: torch.Tensor,
           top_k_weights: torch.Tensor,
       ) -> torch.Tensor:
           ...
   ```
   - Note: `get_use_experts_implementation_args` parses the class AST directly,
     so `@use_experts_implementation` (or `@use_experts_implementation(...)`
     with keyword flags `is_concatenated`, `is_transposed`, `has_bias`,
     `has_gate`) must appear directly on the class definition.
   - Parameter attributes MUST be named `down_proj` and either `gate_up_proj`
     (when `has_gate=True`) or `up_proj` (when `has_gate=False`).
   - Do **not** override `forward` in a way that drops `(hidden_states,
     top_k_index, top_k_weights)` or skips `@use_experts_implementation`.
2. **2D checkpoints**: If the checkpoint on disk stores unfused 2D weights
   (`experts.{i}.{gate,up,down}_proj.weight`), still implement `<Arch>Experts`
   with `@use_experts_implementation` + 3D weights and register 2D load/save
   mappings in `src/llmcompressor/modeling/moe/conversion_mappings.py` (or use
   `patch_moe_mappings` per checkpoint; see
   [add-moe-support.md](../../docs/developer-tutorials/add-moe-support.md)).
3. **Fallback**: Only create a `LinearExperts2D` subclass in
   `src/llmcompressor/modeling/moe/` if the architecture's expert formulation
   cannot satisfy `FusedExpertsProtocol`.

### 4. Offloading, Calibration & Tracing Readiness
Verify every item below so `load_context()`, sequential `oneshot()`, and
`save_pretrained()` work without runtime failures:
- **`_no_split_modules`**: Set `_no_split_modules = ["<Arch>DecoderLayer"]` on
  `<Arch>PreTrainedModel` so sequential calibration and offloading auto-resolve
  decoder boundaries.
- **`_skip_keys_device_placement`**: Set `_skip_keys_device_placement = ["past_key_values"]`.
- **`base_model_prefix`**: Set `base_model_prefix = "model"` (matching the
  checkpoint prefix) and implement `get_input_embeddings`,
  `set_input_embeddings`, `get_output_embeddings`, `set_output_embeddings`,
  `get_decoder`, `set_decoder`, and `_tied_weights_keys` (if tied).
- **Meta-tensor / offload-safe `_init_weights` & `forward`**:
  - Ensure `_init_weights` handles `meta` tensors safely and `post_init()` is
    called at the end of `__init__`.
  - Never decorate any `forward` method with `@torch.inference_mode()` (causes
    in-place calibration/GPTQ writeback crashes). Use `torch.no_grad()` only
    where needed.
  - Derive new tensor devices from activation tensors (e.g.,
    `hidden_states.device`) or `get_execution_device(module)`, not directly
    from offloaded parameter `.device`.
  - Keep `DecoderLayer` and `Model.forward` FX-traceable (`trace_subgraphs`
    compatible): avoid dynamic Python control flow or in-place state mutations
    on traced proxy tensors.
  - Ensure all custom layers initialize in the checkpoint/model `dtype` (e.g.,
    `bfloat16`), avoiding unexpected `float32` mismatches on norms or gates.

---

## Step 3 — Shrinking Large Checkpoints (When Full Weights Exceed VRAM)

### Option A: Tiny Verification Model (Required for Unit / CI Tests)
In `tests/llmcompressor/modeling/test_<arch>.py`, construct a tiny
`<Arch>Config` (e.g., 2–4 hidden layers, small `hidden_size` / `intermediate_size`,
few experts) and save it to `tmp_path`. See
[create-tiny-model/SKILL.md](create-tiny-model/SKILL.md) if you also need a
standalone tiny Hub checkpoint.

### Option B: Pruned Checkpoint via `ExpertPruner` / `MagnitudeExpertPruner`
When validating parity locally against **real** checkpoint weights on a
VRAM-constrained GPU, prune MoE experts directly from safetensors on disk:

```python
from compressed_tensors.entrypoints.convert import (
    MagnitudeExpertPruner,
    convert_checkpoint,
)
from llmcompressor.entrypoints.converter import ExpertPruner

# Preferred for 3D stacked experts, routing bias (aux_pattern),
# hash tables (index_pattern), or group-limited routers (n_group > 1):
converter = ExpertPruner.from_pretrained(
    model_name_or_path=MODEL_ID,
    sparsity=0.9,
    metric="magnitude",
    uniform=True,
    router_pattern=r"mlp\.gate\.weight$",
    expert_pattern=r"mlp\.experts\.(gate_up_proj|down_proj)",
)

# Simpler alternative from compressed-tensors for standard 2D/3D MoEs:
# converter = MagnitudeExpertPruner.from_pretrained(
#     model_name_or_path=MODEL_ID,
#     router_pattern=r"mlp\.gate\.weight$",
#     expert_pattern=r"mlp\.experts\.(gate_up_proj|down_proj)",
#     sparsity=0.9,
# )

convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory="./pruned-ckpt",
    converter=converter,
)
```

If even the non-MoE backbone or total layer count still exceeds VRAM, combine
expert pruning with trimming `num_hidden_layers` in the saved `config.json`
(keeping at least one layer of each layer/attention type) and dropping extra
layer weights when testing locally.

---

## Step 4 — Verification & PR Checklist

1. **Test file**: Add `tests/llmcompressor/modeling/test_<arch>.py` from the
   pathway template.
2. **Format & lint**: Run `make style` and `make quality` (see
   [style.md](style.md)).
3. **Run tests** (check `canhazgpu` per [test.md](test.md)):
   ```bash
   python3 -m pytest -v tests/llmcompressor/modeling/test_<arch>.py
   ```
   - All contract tests (round-trip, `DynamicCache` / `QuantizedKVCache`,
     `load_context` MoE linearization, `trace_subgraphs`) must pass without
     `vllm`.
   - Run the parity test in a `vllm`-equipped environment (for `add-vllm-model`)
     or directly in Python (for `add-pytorch-model`).
4. **PR readiness**: Follow `CONTRIBUTING.md` and `.github/PULL_REQUEST_TEMPLATE.md`
   — summarize the architecture, document the parity test results (logprobs /
   logit diffs and contract checks), and list any known limitations or ignored
   layers.