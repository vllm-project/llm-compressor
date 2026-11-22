# Adding Custom Transformers Modeling Support

To run calibrated quantization workflows in LLM Compressor (`oneshot`, `GPTQModifier`, `AWQModifier`, `SmoothQuantModifier`, NVFP4, or KV cache quantization), a model must be loaded as a Hugging Face `transformers` `PreTrainedModel`.

When a new model launches in **vLLM** or as a standalone **`torch.nn.Module`** inference repository before upstream `transformers` adds official support, you can adapt it into a **Custom Modeling Package** under `src/llmcompressor/modeling/<arch>/` using the built-in agent skills and templates.

## When Do You Need a Custom Modeling Package?

Use the table below to choose the right workflow before writing new modeling code:

| Situation | Recommended Path |
| --- | --- |
| Model is already supported in `transformers` and only needs calibration-free or standard quantization | Use `AutoModelForCausalLM` / `*ForConditionalGeneration` with `load_context()` ([`oneshot`](../guides/entrypoints/oneshot.md)) |
| Model is **not** in `transformers` and you only need calibration-free weight quantization (e.g., `FP8_DYNAMIC`, `FP8_BLOCK`, `MXFP4`) | Use [`model_free_ptq`](../guides/entrypoints/model-free-ptq.md) directly on safetensors |
| Model is already in `transformers`, but uses custom MoE modules without `@use_experts_implementation` | Add a `LinearExperts2D` wrapper or 2D conversion mapping ([Adding MoE Calibration Support](add-moe-support.md)) |
| Model is **not** in `transformers` and you need calibrated schemes (`NVFP4`, `GPTQ`, `AWQ`, `SmoothQuant`, KV cache quant, or `REAP`) | Add a **Custom Modeling Package** under `src/llmcompressor/modeling/<arch>/` (this guide) |

## Using the Agent Skills

LLM Compressor ships two agent skills in `.agents/skills/` backed by a shared specification:

- **Porting from vLLM** (`vllm/model_executor/models/<arch>.py`): invoke `.agents/skills/add-vllm-model/SKILL.md`
- **Porting from standalone PyTorch (`torch.nn.Module`)**: invoke `.agents/skills/add-pytorch-model/SKILL.md`
- **Shared contracts and checklist**: `.agents/skills/shared_modeling.md`

Prompt your coding agent with the target architecture name, Hugging Face model ID (for inspecting `config.json` and `model.safetensors.index.json`), and path to the source file:

```text
Use the add-vllm-model skill to add Custom Modeling Package support for <arch>
using <path/to/vllm/model.py> and <org/model-id>.
```

For standalone PyTorch inference definitions (such as author-released `inference/model.py` scripts):

```text
Use the add-pytorch-model skill to add Custom Modeling Package support for <arch>
using <path/to/model.py> and <org/model-id>.
```

## Required Architecture & Contracts

Every Custom Modeling Package must live in `src/llmcompressor/modeling/<arch>/`:

```text
src/llmcompressor/modeling/<arch>/
├── __init__.py
├── configuration_<arch>.py
├── modeling_<arch>.py
└── README.md
```

### 1. Explicit Loading (No Global AutoClass Mutation)

Export `<Arch>Config` and `<Arch>ForCausalLM` (or `<Arch>ForConditionalGeneration`) from `src/llmcompressor/modeling/<arch>/__init__.py`. Do **not** register the model globally in `transformers.AutoConfig` or import it from `src/llmcompressor/modeling/__init__.py`—this avoids import-time side effects and collisions when upstream `transformers` later ships the same `model_type`.

Load the model explicitly with `load_context`:

```python
from llmcompressor.modeling.<arch> import <Arch>ForCausalLM
from llmcompressor.utils import load_context

with load_context(<Arch>ForCausalLM):
    model = <Arch>ForCausalLM.from_pretrained(MODEL_ID)
```

### 2. Checkpoint-Matched Weight Names

`state_dict` keys in `modeling_<arch>.py` must match the **on-disk `.safetensors` checkpoint keys**, not vLLM's internal fused module names. Inspect the vLLM source's `load_weights(...)` and `packed_modules_mapping` (or the checkpoint's `model.safetensors.index.json`) to unwrap `QKVParallelLinear`, `MergedColumnParallelLinear`, `ColumnParallelLinear`, and `RowParallelLinear` back into standard `torch.nn.Linear` projections.

### 3. `PreTrainedConfig`

Subclass `transformers.PreTrainedConfig` in `configuration_<arch>.py` with `model_type = "<arch>"`. Make sure the decoder/text config exposes the standard attributes consumed by `compressed-tensors` and `MoEConfig.from_config`:

- **Attention**: `num_attention_heads`, `num_key_value_heads`, and `head_dim` (or `hidden_size // num_attention_heads`).
- **MoE (if applicable)**: `num_experts` (or `num_local_experts` / `n_routed_experts`), `num_experts_per_tok` (or `top_k_experts`), `hidden_size`, `moe_intermediate_size` (or `intermediate_size`), `hidden_act`, and `dtype`.

### 4. `DynamicCache` for KV Cache Quantization

`compressed_tensors` attaches `QuantizedKVCache` only when `is_cached_attention_module(module)` returns `True`. Ensure every attention module satisfies all three checks:

1. Its class name contains `"attention"` case-insensitively (for example, `<Arch>Attention`).
2. It exposes at least one of `k_proj`, `v_proj`, `qkv_proj`, or `kv_b_proj`.
3. Its `forward` method accepts `past_key_values: Cache | None = None`, updates the cache via `past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)`, and lets the decoder instantiate [`DynamicCache`](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py) when `use_cache=True` and `past_key_values is None`.

Do not replace `DynamicCache` with static tensor buffers (such as `self.kv_cache`) and do not override `_supports_default_dynamic_cache` to `False` on `PreTrainedModel`.

### 5. `@use_experts_implementation` for MoE Linearization

Decorate routed MoE expert classes directly with [`@use_experts_implementation`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/moe.py) from `transformers.integrations.moe`:

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

    def forward(self, hidden_states, top_k_index, top_k_weights):
        ...
```

Because `get_use_experts_implementation_args` inspects the class AST, the decorator must appear directly on the class definition, with `down_proj` and `gate_up_proj` (or `up_proj`) stored as `torch.nn.Parameter`s. If the checkpoint stores 2D per-expert weights on disk, also register 2D load/save conversion mappings (see [Adding MoE Calibration Support](add-moe-support.md)).

### 6. Offloading, Calibration & Tracing Safety

- Set `_no_split_modules = ["<Arch>DecoderLayer"]` and `_skip_keys_device_placement = ["past_key_values"]` on `<Arch>PreTrainedModel`.
- Implement `get_input_embeddings`, `set_input_embeddings`, `get_output_embeddings`, `set_output_embeddings`, `get_decoder`, and `set_decoder`.
- Keep `_init_weights` safe for `meta` tensors and never decorate `forward` with `@torch.inference_mode()` (which breaks in-place GPTQ/calibration updates).
- Derive dynamic tensor devices from activations (`hidden_states.device`) or `get_execution_device(module)` rather than offloaded parameters.

## Testing & Shrinking Oversized Checkpoints

Copy the matching parity test template into `tests/llmcompressor/modeling/test_<arch>.py`:

- **vLLM pathway**: `.agents/skills/add-vllm-model/templates/test_vllm_parity.py`
- **PyTorch pathway**: `.agents/skills/add-pytorch-model/templates/test_pytorch_parity.py`

### Automated Unit & Parity Tests

The committed test suite uses a **Tiny Verification Model** (2–4 layers, small hidden dims saved to `tmp_path`) so CI runs fast without downloading multi-hundred-GB checkpoints:

1. **Unconditional contract tests** (run in `pytest` without `vllm`):
   - Save/load round-trip via `load_context(<Arch>ForCausalLM)`
   - `DynamicCache` prefill/decode parity and `QuantizedKVCache` attachment (`is_cached_attention_module`, `k_scale`, `v_scale`)
   - `@use_experts_implementation` linearization, `moe_calibration_context()`, and `repack_moe`
   - Sequential subgraph tracing via `trace_subgraphs`
2. **Parity test**:
   - **vLLM**: gated behind `@requires_gpu(1)` and `pytest.importorskip("vllm")`, comparing prompt logprobs and greedy generated tokens between `vllm.LLM` and the Custom Modeling Package on the same checkpoint.
   - **PyTorch**: compares prefill logits and cached decode outputs directly against the reference `torch.nn.Module`.

### Shrinking Real Checkpoints with `ExpertPruner` / `MagnitudeExpertPruner`

When validating locally against real checkpoint weights that do not fit in VRAM, prune experts on disk before running parity checks:

```python
from compressed_tensors.entrypoints.convert import (
    MagnitudeExpertPruner,
    convert_checkpoint,
)
from llmcompressor.entrypoints.converter import ExpertPruner

# Use ExpertPruner for 3D stacked experts, routing bias, hash tables, or grouped routing
converter = ExpertPruner.from_pretrained(
    model_name_or_path=MODEL_ID,
    sparsity=0.9,
    metric="magnitude",
    uniform=True,
    router_pattern=r"mlp\.gate\.weight$",
    expert_pattern=r"mlp\.experts\.(gate_up_proj|down_proj)",
)

# Or MagnitudeExpertPruner from compressed-tensors for simpler checkpoints:
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

If even the pruned checkpoint's dense backbone or layer count exceeds your GPU memory, pair expert pruning with reducing `num_hidden_layers` in `config.json` (preserving at least one layer of every attention/layer variant) for local verification.

## Submitting a Pull Request

Before opening a PR (see [CONTRIBUTING.md](https://github.com/vllm-project/llm-compressor/blob/main/CONTRIBUTING.md)):

1. Format and lint all code (do not add `pyproject.toml` exclusions):
   ```bash
   make style
   make quality
   ```
2. Run the modeling and contract tests (with `canhazgpu` if available on your machine):
   ```bash
   python3 -m pytest -v tests/llmcompressor/modeling/test_<arch>.py
   ```
3. Run the parity test in a `vllm`-enabled environment (for vLLM ports) or in PyTorch (for `torch.nn.Module` ports), and include the parity results (tested checkpoint/tiny config, logprobs or logit diff, and calibration smoke results) in your PR description.
4. Keep `src/llmcompressor/modeling/<arch>/README.md` clear and concise with upstream provenance and any known quirks.