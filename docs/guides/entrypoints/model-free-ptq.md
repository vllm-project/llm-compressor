# model_free_ptq

`model_free_ptq` is a PTQ entrypoint for **data-free quantization schemes** that operates directly on safetensors checkpoint files without requiring a Hugging Face model definition or loading the model through `transformers`.

## When to Use

Use `model_free_ptq` when:

- Your quantization scheme is **data-free** (e.g. FP8 dynamic, FP8 block, NVFP4A16, MXFP4/MXFP8)
- The model **does not have a Hugging Face transformers definition** (e.g. a newly released model not yet in transformers)
- `oneshot` **fails** for your model

For schemes that require calibration data (GPTQ, AWQ, SmoothQuant, static activation quantization), use [`oneshot`](oneshot.md) instead.

## Basic Usage

```python
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="meta-llama/Meta-Llama-3-8B-Instruct",
    save_directory="Meta-Llama-3-8B-Instruct-FP8-BLOCK",
    scheme="FP8_BLOCK",
    ignore=["lm_head"],
    device="cuda:0",
)
```

## QuantizationConfig

Use `config` instead of `scheme` when different data-free schemes should be
applied to different layers, or when the output should include an optional KV
cache quantization scheme. `scheme` and `config` are mutually exclusive. Each
config group provides its own targets and quantization scheme; when groups
overlap, the first matching group wins.

```python
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, MXFP4
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="inference-optimization/GLM-5.2-0.8B-A0.8B",
    save_directory="GLM-5.2-0.8B-A0.8B-MXFP4-FP8-BLOCK",
    config=QuantizationConfig(
        config_groups={
            "attention": QuantizationScheme(
                targets=[r"re:.*self_attn.*"],
                weights=FP8_BLOCK["weights"],
                input_activations=FP8_BLOCK["input_activations"],
            ),
            "mlp": QuantizationScheme(
                targets=[r"re:.*mlp.*"],
                weights=MXFP4["weights"],
                input_activations=FP8_BLOCK["input_activations"],
            ),
        }
    ),
    ignore=["lm_head", r"re:.*router.*"],
)
```

## Multi-GPU Execution

Pass a list of devices to use multiple GPUs:

```python
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="meta-llama/Meta-Llama-3-8B-Instruct",
    save_directory="Meta-Llama-3-8B-Instruct-FP8-BLOCK",
    scheme="FP8_BLOCK",
    ignore=["lm_head"],
    device=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    max_workers=4,
)
```

When `device=None`, all visible CUDA devices are selected automatically. 

## How It Works

`model_free_ptq` processes each `.safetensors` file in the checkpoint independently, without ever loading the full model into memory as a `torch.nn.Module`. For each file:

1. **Validate** — check that all quantizable tensors can be quantized with the given scheme
2. **Initialize** — create a minimal `torch.nn.Linear` module for each weight tensor
3. **Calibrate** — compute scale and zero point directly from the weight tensor (data-free)
4. **Estimate** — validate on the meta device and estimate each shard's peak memory
5. **Compress** — schedule each shard on a suitable device and call `compress_module` from `compressed-tensors` to pack/quantize the weights
6. **Save** — write the compressed tensors back to disk

After all files are processed, the safetensors index and model config are updated with the quantization metadata.

Multiple files can be processed in parallel using the `max_workers` argument.

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model_stub` | `str \| PathLike` | — | HuggingFace model ID or path to a local directory containing safetensors files |
| `save_directory` | `str \| PathLike` | — | Directory to save the quantized checkpoint |
| `scheme` | `QuantizationScheme \| str \| None` | `None` | One quantization scheme to apply. Mutually exclusive with `config` |
| `config` | `QuantizationConfig \| None` | `None` | One or more data-free config groups, optionally including `kv_cache_scheme`; mutually exclusive with `scheme` |
| `ignore` | `Iterable[str]` | `()` | Module names or regex patterns to skip. Modules ending in `"norm"` are always ignored automatically |
| `max_workers` | `int` | `1` | Upper bound on concurrent worker threads for processing safetensors shards. Effective concurrency may be lower when GPU memory is tight |
| `device` | `str \| torch.device \| list[str \| torch.device] \| None` | `None` | Device or devices to use. A list enables multi-GPU shard scheduling; `None` automatically selects all visible CUDA devices, or CPU when no accelerator is available |
| `converter` | `Converter \| None` | `None` | Optional `compressed-tensors` converter to apply before quantization, e.g. to convert modelopt-format checkpoints to compressed-tensors format |

## Standard Flow (Non-Microscale Schemes)

For schemes without a global scale (e.g. `FP8_BLOCK`, `FP8_DYNAMIC`), call `model_free_ptq` directly:

```python
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="unsloth/Kimi-K2-Thinking-BF16",
    save_directory="Kimi-K2-Thinking-FP8-BLOCK",
    scheme="FP8_BLOCK",
    ignore=[
        "re:.*gate$",
        "lm_head",
        "re:.*kv_a_proj_with_mqa$",
        "re:.*q_a_proj$",
        "model.embed_tokens",
    ],
    max_workers=15,
    device="cuda:0",
)
```

## Microscale Flow (NVFP4A16)

NVFP4 weight-only quantization requires a **global scale** that is fused across
related weight groups (e.g. qkv projections, gate/up projections).
`model_free_ptq` handles this fusion directly, so no preprocessing step is
required — run it just like the non-microscale schemes above:

```python
from llmcompressor import model_free_ptq
from compressed_tensors.quantization import preset_name_to_scheme

nvfp4_scheme = preset_name_to_scheme("NVFP4A16", targets=["Linear"])
nvfp4_scheme.weights.observer = "nvfp4_expanded_mse"

model_free_ptq(
    model_stub="unsloth/Kimi-K2-Thinking-BF16",
    save_directory="Kimi-K2-Thinking-NVFP4A16",
    scheme=nvfp4_scheme, # "NVFP4A16"
    ignore=[
        "re:.*gate$",
        "lm_head",
        "re:.*kv_a_proj_with_mqa$",
        "re:.*q_a_proj$",
        "model.embed_tokens",
    ],
    max_workers=15,
    device="cuda:0",
)
```

Note: The above setup changes the observer to our recommended NVFP4/NVFP4A16 observer though you can use the default observer if desired

## Ignoring Layers

The `ignore` argument accepts module name strings or regex patterns prefixed with `re:`. Modules whose names end in `"norm"` are automatically ignored regardless of the `ignore` list.

```python
ignore=[
    "lm_head",            # exact name match
    "re:.*gate$",         # regex: any module ending in "gate"
    "model.embed_tokens", # exact name match
]
```

## Supported Schemes

`model_free_ptq` supports any data-free weight quantization scheme. Common presets:

| Scheme | Description |
|--------|-------------|
| `FP8_DYNAMIC` | FP8 weights with dynamic per-token activation quantization |
| `FP8_BLOCK` | FP8 weights with block-wise scaling (Blackwell-optimized) |
| `NVFP4A16` | NVFP4 weight-only quantization with FP8 group scales and a global scale |
| `MXFP4/MXFP8` | MXFP4 or MXFP8 quantization with MX-format microscales |

Note: Many of these schemes, such as NVFP4 and MXFP4 may potentially lead to improved recovery when applied with a calibration algorithm that requires data, such as GPTQ. Consider comparing performance using oneshot.
For the full list of supported schemes and formats, see [Compression Schemes](../compression_schemes.md).
