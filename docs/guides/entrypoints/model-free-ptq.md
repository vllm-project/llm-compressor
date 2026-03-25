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

## How It Works

`model_free_ptq` processes each `.safetensors` file in the checkpoint independently, without ever loading the full model into memory as a `torch.nn.Module`. For each file:

1. **Validate** — check that all quantizable tensors can be quantized with the given scheme
2. **Initialize** — create a minimal `torch.nn.Linear` module for each weight tensor
3. **Calibrate** — compute scale and zero point directly from the weight tensor (data-free)
4. **Compress** — call `compress_module` from `compressed-tensors` to pack/quantize the weights
5. **Save** — write the compressed tensors back to disk

After all files are processed, the safetensors index and model config are updated with the quantization metadata.

Multiple files can be processed in parallel using the `max_workers` argument.

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model_stub` | `str \| PathLike` | — | HuggingFace model ID or path to a local directory containing safetensors files |
| `save_directory` | `str \| PathLike` | — | Directory to save the quantized checkpoint |
| `scheme` | `QuantizationScheme \| str` | — | Quantization scheme to apply. Can be a preset string (e.g. `"FP8_BLOCK"`, `"NVFP4A16"`) or a `QuantizationScheme` object |
| `ignore` | `Iterable[str]` | `()` | Module names or regex patterns to skip. Modules ending in `"norm"` are always ignored automatically |
| `max_workers` | `int` | `1` | Number of parallel worker threads for processing safetensors files |
| `device` | `str \| torch.device \| None` | `None` | Device to use for quantization. Defaults to GPU if available, otherwise CPU |
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

## Microscale Flow (NVFP4)

NVFP4 requires a **global scale** that is fused across related weight groups (e.g. qkv projections, gate/up projections). For this fusion to work correctly, the weights of each fused group must reside in the **same safetensors shard**.

Standard model checkpoints often split these weights across different shards. To fix this, run the `reindex_fused_weights` CLI tool first to reorganize the checkpoint:

```bash
llmcompressor.reindex_fused_weights \
    unsloth/Kimi-K2-Thinking-BF16 \
    Kimi-K2-Thinking-BF16-reindexed \
    --num_workers=10
```

Then run `model_free_ptq` on the reindexed checkpoint:

```python
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="Kimi-K2-Thinking-BF16-reindexed",
    save_directory="Kimi-K2-Thinking-NVFP4A16",
    scheme="NVFP4A16",
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

!!! note
    Reindexing is only required for **NVFP4**, which uses a global scale. MXFP4 does not use a global scale and does not require reindexing.

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
