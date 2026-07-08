# Shared Quantization Skill Documentation

This file contains common logic shared between FP8 and NVFP4 quantization skills.

## Gathering Model Information

When generating a quantization example, collect the following information:

1. **MODEL_ID** — HuggingFace model ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`)

2. **Model type** — dense, MoE, or multimodal (vision/audio)

3. **Architecture class (VLM models only, REQUIRED)** — for any multimodal model, fetch the model's `config.json` from HuggingFace to get the exact architecture class before writing the file:
   - Fetch `https://huggingface.co/{MODEL_ID}/raw/main/config.json` and read the `architectures` field (e.g. `["Llama4ForConditionalGeneration"]`)
   - Use that class directly in the import and `from_pretrained` call instead of `AutoModelForCausalLM`
   - To check if the class is in standard `transformers`, run: `python -c "from transformers import <ClassName>"`. If it imports successfully, no `trust_remote_code` needed. If it fails with `ImportError`, add `trust_remote_code=True` to both `from_pretrained` calls.
   - Do **not** use `AutoModelForCausalLM` for VLM models — always use the specific class
   - MoE models can use `AutoModelForCausalLM` like dense models

4. **Gate/router layer names (MoE models only, REQUIRED)** — for any MoE model you **must** determine the exact names of the gate/router layers before writing the file. Do one of the following:
   - Ask the user: *"What are the gating/routing layer names in this model? (e.g. `mlp.gate`, `router`)"*
   - Or inspect the model config: fetch `https://huggingface.co/{MODEL_ID}/raw/main/config.json`, search HuggingFace, or look at existing examples in `examples/` for architecture details to derive the correct regex pattern.
   - Do **not** write the file until the gate/router layer pattern is confirmed. These layers produce logits that control expert routing — quantizing them degrades routing decisions and causes accuracy loss.
   - Once you have the layer name, construct the regex as follows:
     - Escape any literal dots: `mlp.gate` → `mlp\.gate`
     - Anchor with `$` to prevent partial matches on layers sharing the same prefix (e.g. `mlp\.gate$` matches `mlp.gate` but not `mlp.gate_proj`)
     - Prepend `.*` to match at any depth: `"re:.*mlp\.gate$"`
     - **Always use a raw string (`r"..."`) for any ignore pattern containing a backslash** to avoid Python `SyntaxWarning: invalid escape sequence`

## Model-Type Adjustments

**Rule:** Always ignore `lm_head`, any vision tower layers, and any gating/routing layers. The exact regex patterns must be derived from the model's actual architecture. All other layers (e.g. `model.embed_tokens`, attention layers, projectors) are only added to `ignore` when the user explicitly requests it.

### Dense models
`ignore=["lm_head"]` is sufficient.

### MoE models
**Required:** Always add gate/router layers to `ignore`. These control expert routing — quantizing them causes routing degradation. The exact pattern must come from the model's actual architecture.

Always wrap the load in `load_context` using `AutoModelForCausalLM`:
```python
from transformers import AutoModelForCausalLM
from llmcompressor.utils import load_context

with load_context(AutoModelForCausalLM):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
```

Common MoE gating patterns (always verify against the model's actual layer names before using):
- Qwen MoE: `"re:.*mlp.gate$"`, `"re:.*shared_expert_gate.*"`
- Llama4 / Gemma4 MoE: `"re:.*router"`
- Laguna-M.1: `"re:.*mlp\.gate$"` (targets `mlp.gate` routing layers; avoids false match on `gate_proj`)
- Trinity: `"re:.*mlp.router.*"`

If the gate/router layer is an `nn.Parameter` (not `nn.Linear`) it will not be targeted by `targets="Linear"`, but it should still be explicitly listed in `ignore` to document intent and guard against future changes.

Some deepseek-like architectures use an `attn.indexer` and `attn.indexer.compressor`. These weights are sensitive to quantization and should be ignored.

### Multimodal (vision / audio)
- Use `AutoProcessor` instead of `AutoTokenizer`
- Always fetch the model's `config.json` to get the specific class from the `architectures` field (e.g. `Gemma4ForConditionalGeneration`, `Llama4ForConditionalGeneration`) — **never use `AutoModelForCausalLM` for VLMs**
- Import and use that class directly; wrap in `load_context`:
  ```python
  from transformers import Llama4ForConditionalGeneration  # example
  from llmcompressor.utils import load_context

  with load_context(Llama4ForConditionalGeneration):
      model = Llama4ForConditionalGeneration.from_pretrained(MODEL_ID)
  ```
- Add `trust_remote_code=True` if the class is not in standard transformers
- Always ignore vision tower layers using a single regex that covers the model's actual layer names (verify against the architecture — one pattern is sufficient since regex search matches substrings):
  - Vision: `["re:.*visual.*", ".*vision_tower.*"]`
  - Audio: `"re:audio.*"` (matches `audio_tower.*` etc.)

## Sample Generation

Check the model's parameter count from its name or by fetching `config.json` (look for a size indicator in the model ID such as `70B`, `72B`, `405B`). If the model exceeds **70B parameters**, omit the entire sample generation block (from `dispatch_model` through the closing `print`) to avoid OOM.

## File Naming and Styling

After writing the file, always run `make style` to format it correctly.
