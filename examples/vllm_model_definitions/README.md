# vLLM Model Definition Loading

This folder shows model loading using vLLM model definitions only (no
`transformers.AutoModel*` loading).

## What changed in llmcompressor

`llmcompressor` sequential tracing support checks were refactored to use vLLM model
architectures from:

- `vllm.model_executor.models.ModelRegistry.get_supported_archs()`

instead of task/model mapping constants from `transformers`.

## Why this works

- `ModelRegistry` is built from vLLM's model folder (`vllm/model_executor/models/*`).
- Loading classes through `ModelRegistry.models[arch].load_model_cls()` imports the
  actual vLLM model definition module for that architecture.
- Runtime loading through `vllm.LLM(...)` uses vLLM's model resolution and model
  executor stack.

## Scripts

### 1) Check if an architecture is in vLLM registry

```bash
python3 check_registry_support.py --architecture LlamaForCausalLM
```

### 2) Load model class from vLLM registry/model folder

```bash
python3 load_vllm_model_class.py --architecture LlamaForCausalLM
```

This script prints the loaded class and module path, proving the model definition was
loaded from vLLM code.

### 3) Load and run model with vLLM runtime

```bash
python3 load_with_vllm_runtime.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

## Note on compression pipelines

This folder demonstrates model-definition loading and runtime loading with vLLM.
If you are using `oneshot`, keep in mind that broader compression entrypoints may
still include `transformers`-based utilities in other parts of the stack.

## KV-cache status

KV-cache quantization paths are currently disabled in llmcompressor and
`kv_cache_scheme` is ignored for now.
