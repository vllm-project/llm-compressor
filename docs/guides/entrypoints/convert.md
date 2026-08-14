# convert_checkpoint

`convert_checkpoint` is an entrypoint in `compressed-tensors` for converting a model checkpoint from one quantization format to another. It operates entirely within the `compressed-tensors` library, working directly on safetensors files and updating the `quantization_config` in `config.json`.

## When to Use

Use `convert_checkpoint` when you have a checkpoint quantized in a format (e.g. ModelOpt, AutoAWQ) and need to convert it to a different format or back to dense. Common cases:

- A checkpoint published in ModelOpt or AutoAWQ format that needs to be expressed in compressed-tensors format
- A compressed-tensors checkpoint that needs to be re-quantized under a different scheme, requiring a round-trip through dense weights first

Converters can also be passed inline to [`model_free_ptq`](model-free-ptq.md) via its `converter` argument, which applies the conversion as part of the quantization run without writing an intermediate checkpoint to disk.

## Basic Usage

```python
from compressed_tensors.entrypoints.convert import convert_checkpoint, FP8BlockDequantizer

convert_checkpoint(
    model_stub="deepseek-ai/DeepSeek-V3.2",
    save_directory="DeepSeek-V3.2-bf16",
    converter=FP8BlockDequantizer(
        targets=[
            r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
            r"re:.*self_attn.*\.(kv_b|o|q_a|q_b)_proj$",
        ],
    ),
    max_workers=4,
)
```

## How It Works

`convert_checkpoint` accepts a `model_stub` (HuggingFace model ID or local path), a `save_directory`, a `converter`, and an optional `max_workers` count for parallelism. It processes the checkpoint in four stages:

1. **Resolve** — collect all safetensors files and build a weight map from weight name to shard file
2. **Plan** — compute an inverse weight map so each output shard knows which source files to load (converters may require tensors from other shards as dependencies, declared via `converter.get_dependencies()`)
3. **Validate** — run `converter.validate()` on each shard to catch incompatibilities before the long-running conversion begins
4. **Convert** — run `converter.process()` on each shard, write converted tensors to `save_directory`, and update `config.json` with the output of `converter.create_config()`

All non-safetensors files (tokenizer, config, etc.) are copied to `save_directory` unchanged.

## Available Converters

### `ModelOptNvfp4Converter`

Converts NVIDIA ModelOpt NVFP4 checkpoints to the compressed-tensors NVFP4 format. Renames tensors, inverts global scales, and produces a `QuantizationConfig` for the output checkpoint.

**Use when:** your NVFP4 checkpoint was produced by ModelOpt. 

Accepts `targets` and `ignore` as module name strings or `re:`-prefixed regex patterns, plus an optional `kv_cache_scheme` for KV cache scales. Layers not listed in `targets` are left unchanged.

```python
from compressed_tensors.entrypoints.convert import ModelOptNvfp4Converter

converter = ModelOptNvfp4Converter(
    targets=["re:.*mlp.*(gate_up|gate|up|down)_proj$"],
)
```

**Tensor mapping (ModelOpt → compressed-tensors):**

| ModelOpt tensor | compressed-tensors tensor | Transformation |
|-----------------|--------------------------|----------------|
| `input_scale` | `input_global_scale` | Inverted (`1/x`) |
| `weight` | `weight_packed` | Renamed |
| `weight_scale` | `weight_scale` | Unchanged |
| `weight_scale_2` | `weight_global_scale` | Inverted (`1/x`) |

---

### `AutoAWQConverter`

Converts AutoAWQ checkpoints (4-bit) to compressed-tensors W4A16 `pack_quantized` format. Unpacks the `qweight` and `qzeros` tensors, reorders the bit layout, and shifts to a signed integer representation.

**Use when:** your checkpoint was produced by AutoAWQ.

The easiest way to construct this converter is via `from_pretrained`, which reads the AWQ config directly from the checkpoint:

```python
from compressed_tensors.entrypoints.convert import AutoAWQConverter

converter = AutoAWQConverter.from_pretrained("TheBloke/Llama-2-7B-AWQ")
```

---

### `FP8BlockDequantizer`

Dequantizes FP8 block-quantized weights back to a dense floating-point format (default: `bfloat16`). For each targeted layer, combines the `weight` tensor with its paired `weight_scale_inv` to reconstruct full-precision values. `create_config()` returns `None`, so `quantization_config` is removed from the output `config.json`.

**Use when:** you want to convert a checkpoint that was quantized with `FP8_BLOCK` (e.g. `deepseek-ai/DeepSeek-V3.2`) to dense weights.

Accepts `targets` and `ignore` as module name strings or `re:`-prefixed regex patterns. The `weight_block_size` (default `(128, 128)`) must match the block size used during original quantization. Set `dtype` to control the output precision (default `torch.bfloat16`).

```python
from compressed_tensors.entrypoints.convert import FP8BlockDequantizer

converter = FP8BlockDequantizer(
    targets=[
        r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
        r"re:.*self_attn.*\.(kv_b|o|q_a|q_b)_proj$",
    ],
)
```

---

### `CompressedTensorsDequantizer`

Dequantizes any checkpoint stored in compressed-tensors format (e.g. W4A16, INT8) back to a dense floating-point format. Reads the `quantization_config` from the source checkpoint and uses the appropriate decompressor per scheme. `create_config()` returns `None`, so `quantization_config` is removed from the output `config.json`.

**Use when:** you have a compressed-tensors checkpoint and need to convert it to dense weights — for example, `moonshotai/Kimi-K2.6` (W4A16) before re-quantizing to NVFP4.

Accepts a `model_stub` pointing to the source checkpoint (used to read the quantization config), `ignore` patterns for layers to leave in their current compressed state, and a `dtype` for the output precision (default `torch.bfloat16`).

```python
from compressed_tensors.entrypoints.convert import CompressedTensorsDequantizer

converter = CompressedTensorsDequantizer(
    model_stub="moonshotai/Kimi-K2.6",
    ignore=[
        "re:.*mlp.gate$",
        "re:.*lm_head",
        "re:.*self_attn.*",
    ],
)
```

---

## Examples

### Convert ModelOpt NVFP4 and apply FP8_BLOCK

Convert an NVFP4 ModelOpt checkpoint to compressed-tensors format, then apply FP8_BLOCK quantization to the attention layers using `model_free_ptq`. Passing the converter directly to `model_free_ptq` avoids writing an intermediate checkpoint.

```python
from compressed_tensors.entrypoints.convert import ModelOptNvfp4Converter
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="nvidia/DeepSeek-R1-NVFP4",
    save_directory="DeepSeek-R1-NVFP4-FP8-BLOCK",
    scheme=QuantizationScheme(
        **FP8_BLOCK,
        targets=["re:.*self_attn.(kv_a_proj_with_mqa|q_a_proj|o_proj|q_b_proj).*"],
    ),
    max_workers=8,
    device="cuda:0",
    converter=ModelOptNvfp4Converter(
        targets=["re:.*mlp.*(gate_up|gate|up|down)_proj$"],
    ),
)
```

Full example: [`examples/model_free_ptq/deepseek_r1_nvfp4_fp8_block.py`](https://github.com/vllm-project/llm-compressor/tree/main/examples/model_free_ptq/deepseek_r1_nvfp4_fp8_block.py)

---

### Dequantize FP8_BLOCK to bfloat16, then re-quantize

Convert an FP8_BLOCK checkpoint to bfloat16, then apply a different quantization scheme using `oneshot`.

```python
from compressed_tensors.entrypoints.convert import FP8BlockDequantizer, convert_checkpoint
from llmcompressor import oneshot

# Step 1: convert to bfloat16
convert_checkpoint(
    model_stub="deepseek-ai/DeepSeek-V3.2",
    save_directory="DeepSeek-V3.2-bf16",
    converter=FP8BlockDequantizer(
        targets=[
            r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
            r"re:.*self_attn.*\.(kv_b|o|q_a|q_b)_proj$",
        ],
    ),
    max_workers=4,
)

# Step 2: re-quantize with oneshot
oneshot(
    model="DeepSeek-V3.2-bf16",
    recipe=...,
)
```

Full example: [`examples/disk_offloading/deepseek_v32_example.py`](https://github.com/vllm-project/llm-compressor/tree/main/examples/disk_offloading/deepseek_v32_example.py)

---

### Dequantize a compressed-tensors W4A16 checkpoint

Dequantize `moonshotai/Kimi-K2.6` (published in W4A16 compressed-tensors format) to bfloat16 before re-quantizing to NVFP4.

```python
from compressed_tensors.entrypoints.convert import CompressedTensorsDequantizer, convert_checkpoint

convert_checkpoint(
    model_stub="moonshotai/Kimi-K2.6",
    save_directory="Kimi-K2.6-bf16",
    converter=CompressedTensorsDequantizer(
        "moonshotai/Kimi-K2.6",
        ignore=[
            "re:.*mlp.gate$",
            "re:.*lm_head",
            "re:.*self_attn.*",
            "re:.*embed_tokens$",
        ],
    ),
    max_workers=4,
)
```

Full example: [`examples/disk_offloading/kimi_k26_example.py`](https://github.com/vllm-project/llm-compressor/tree/main/examples/disk_offloading/kimi_k26_example.py)

---

## The Converter Protocol

All converters implement the `Converter` protocol defined in `compressed_tensors`. You can implement a custom converter by defining these four methods:

```python
from typing import Optional
from compressed_tensors.quantization import QuantizationConfig
import torch

class MyConverter:
    def validate(self, tensors: dict[str, torch.Tensor]) -> None:
        """
        Called once per shard before conversion begins. Raise ValueError or
        log warnings if the shard is incompatible with this converter.
        """
        ...

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Transform tensor names and values. Called once per shard.
        May rename, dequantize, repack, or reorder tensors.
        """
        ...

    def create_config(self) -> Optional[QuantizationConfig]:
        """
        Return the QuantizationConfig to write into config.json, or None to
        remove quantization_config from the output config.
        """
        ...

    def get_dependencies(self, weight_name: str) -> set[str]:
        """
        Return any additional weight names that must be loaded alongside
        weight_name to process it correctly (e.g. paired scale tensors that
        live in a different shard).
        """
        ...
```

`convert_checkpoint` uses `get_dependencies` to build a complete load plan before processing, so converters can safely access cross-shard tensors in `process`.
