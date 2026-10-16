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
)
```

## How It Works

`convert_checkpoint` accepts a `model_stub` (HuggingFace model ID or local path), a `save_directory`, a `converter`, and an optional `max_workers` count for parallelism (`"auto"` by default, which picks the count from the number of shards, their estimated memory, free device memory, and available CPUs). It processes the checkpoint in four stages:

1. **Resolve** — collect all safetensors files and build a weight map from weight name to shard file
2. **Plan** — compute an inverse weight map so each output shard knows which source files to load (converters may require tensors from other shards as dependencies, declared via `converter.get_dependencies()`)
3. **Validate** — run `converter.validate()` on meta tensors for each shard to catch incompatibilities before the long-running conversion begins
4. **Convert** — run `converter.process()` on each shard, write converted tensors to `save_directory`, and update `config.json` using `converter.update_config(config)`

Non-safetensors files, including tokenizer files, are copied to `save_directory`. The pipeline then updates the model config and safetensors index. An existing compressed-tensors quantization config is passed to `update_config`; an external format's config is replaced with the converter's output. Returning `None` removes `quantization_config`.

Pass a list of converters to apply them in order. Each converter receives the previous converter's validation, tensor, and quantization config outputs. The optional `device` argument selects the conversion device; use `device="cpu", max_workers=1` for a CPU run.

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

Converts AutoAWQ GEMM checkpoints (4-bit) to compressed-tensors W4A16 `pack-quantized` format. Unpacks the `qweight` and `qzeros` tensors, reorders the bit layout, and shifts to a signed integer representation. Other bit widths and AutoAWQ layouts such as GEMV are not supported by this converter.

**Use when:** your checkpoint was produced by AutoAWQ.

The easiest way to construct this converter is via `from_pretrained`, which reads the AWQ config directly from the checkpoint:

```python
from compressed_tensors.entrypoints.convert import AutoAWQConverter

converter = AutoAWQConverter.from_pretrained("TheBloke/Llama-2-7B-AWQ")
```

For a complete conversion and verification workflow, see the [AutoAWQ walkthrough](#autoawq-gemm-w4a16-walkthrough).

---

### `FP8BlockDequantizer`

Dequantizes FP8 block-quantized weights back to a dense floating-point format (default: `bfloat16`). For each targeted layer, combines the `weight` tensor with its paired `weight_scale_inv` to reconstruct full-precision values. `update_config(config)` returns `None`, so `quantization_config` is removed from the output `config.json`.

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

Dequantizes any checkpoint stored in compressed-tensors format (e.g. W4A16, INT8) back to a dense floating-point format. Reads the `quantization_config` from the source checkpoint and uses the appropriate decompressor per scheme. `update_config(config)` returns `None`, so `quantization_config` is removed from the output `config.json`.

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

### AutoAWQ GEMM W4A16 Walkthrough

This walkthrough starts from an already-quantized checkpoint. It preserves the source quantization parameters while changing their storage representation; no calibration dataset or AutoAWQ installation is needed.

#### Prepare the source checkpoint

Follow the [installation guide](../../getting-started/install.md) and run the commands below from a checkout of this repository. The example uses the Converter interface in compressed-tensors 0.19 or newer. The optional CPU loading test requires Transformers 5.16 or newer.

Download a fixed revision of the public [Qwen2.5-0.5B-Instruct-AWQ checkpoint](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ):

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct-AWQ \
    --revision db09cd27ead7fee40cdee309693cf83601b9c899 \
    --include '*.json' --include '*.safetensors' --include merges.txt \
    --local-dir ./qwen2.5-0.5b-awq
```

Its `config.json` specifies `quant_method="awq"`, `version="gemm"`, `bits=4`, `group_size=128`, and `zero_point=true`. This worked example requires asymmetric GEMM weights and a positive group size that divides each layer's input dimension. Retain the source checkpoint for the numerical comparison and choose a new output directory.

#### Understand the tensor representation

Let `K` be a linear layer's input features, `N` its output features, and `G = K / group_size`. For the selected checkpoint, `N` is divisible by eight.

| Source AutoAWQ tensor | Source shape / dtype | Output CT tensor | Output shape / dtype |
|-----------------------|----------------------|------------------|----------------------|
| `qweight` | `(K, N/8)`, int32 | `weight_packed` | `(N, ceil(K/8))`, int32 |
| `qzeros` | `(G, N/8)`, int32 | `weight_zero_point` | `(N/8, G)`, int32 |
| `scales` | `(G, N)`, floating point | `weight_scale` | `(N, G)`, same dtype |
| Weight dimensions | Implied by source tensors | `weight_shape` | `(2,)`, int64, values `[N, K]` |

AutoAWQ packs eight unsigned 4-bit values into each int32 along the output feature axis, using the order defined by its [`from_linear` producer](https://github.com/casper-hansen/AutoAWQ/blob/v0.2.9/awq/modules/linear/gemm.py). The converter restores logical feature order, transposes to CT's `(N, K)` convention, and shifts both weights and zero-points by `-8` to the signed range. CT then packs weights along the input feature axis and zero-points along the output feature axis.

For source integer values `Q`, zero-points `Z`, and scales `S`, the reconstructed weight is:

```text
W_source[k, n] = (Q[k, n] - Z[k // group_size, n]) * S[k // group_size, n]
W_CT[n, k]     = ((Q[k, n] - 8) - (Z[k // group_size, n] - 8)) * S[k // group_size, n]
```

The shift cancels, preserving the represented weights. AutoAWQ GEMM stores zero-points directly; do not add one when decoding this source layout.

#### Convert and check numerical consistency

Run `examples/convert_checkpoint/autoawq_example.py` from the repository root:

```bash
python examples/convert_checkpoint/autoawq_example.py \
    --source ./qwen2.5-0.5b-awq \
    --output ./qwen2.5-0.5b-ct \
    --run-inference
```

The conversion itself uses the existing entrypoint:

```python
from compressed_tensors.entrypoints.convert import AutoAWQConverter, convert_checkpoint

convert_checkpoint(
    model_stub="./qwen2.5-0.5b-awq",
    save_directory="./qwen2.5-0.5b-ct",
    converter=AutoAWQConverter.from_pretrained("./qwen2.5-0.5b-awq"),
    device="cpu",
    max_workers=1,
)
```

The example reads tensor headers to locate weights across shards and checks every source `qweight` layer. Its source decoder follows AutoAWQ's producer packing order independently of the CT converter. It verifies:

- Exact agreement of unpacked signed integer weights, zero-points, and transposed scales.
- Exact agreement of reconstructed weights in float32, using the original scale values on both sides. This comparison is against the source **quantized** weights, not the original pre-quantization model.
- Preservation of unchanged tensors, such as biases, embeddings, and normalization weights.
- A matching safetensors weight index and a CT configuration describing compressed, group-wise, asymmetric W4A16 with the source group size.

`convert_checkpoint` writes the output `quantization_config` automatically, including `quant_method="compressed-tensors"`, `format="pack-quantized"`, and `quantization_status="compressed"`. Keep that configuration with the converted tensors and tokenizer files when moving the checkpoint.

#### Verify loading and inference

With `--run-inference`, the example loads the converted checkpoint through Transformers using `CompressedTensorsConfig(dequantize=True)` and `dtype=torch.float32`, rejects missing or unexpected weights, checks that logits are finite, and generates a short greedy completion. This CPU smoke test expands the weights to float32 and checks the saved model can be used for inference. Omit the flag to run only conversion and tensor verification.

For inference with quantized kernels on supported accelerator hardware, load the same output directory with vLLM:

```python
from vllm import LLM, SamplingParams

model = LLM(model="./qwen2.5-0.5b-ct")
outputs = model.generate(
    ["The capital of France is"],
    SamplingParams(temperature=0, max_tokens=16),
)
print(outputs[0].outputs[0].text)
```

See [Deploying with vLLM](../../steps/deploy.md) for environment requirements. Tensor equality establishes preservation of the source weights; different inference kernels and accumulation precision may still produce small differences in logits or generated text.

#### Applying this workflow to QAT outputs

An external QAT workflow must first finalize its inference weights and quantization parameters. An export adapter needs to map the final bit width, signedness, group axis and size, scales, zero-points, and any activation quantization parameters into a compatible CT scheme. A training checkpoint containing fake-quantization wrappers or optimizer state is not directly an AutoAWQ GEMM checkpoint.

For already-quantized integers, adapt the layout and packing as this example does. For floating-point or fake-quantized weights, CT compressors quantize using the supplied parameters before packing, so preserve the external final parameters and verify against the external framework's quantize/dequantize reference. Recomputing scales can change the trained quantization behavior. A runnable QAT workflow and framework-specific QAT adapters are outside this walkthrough's scope.

---

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
)
```

Full example: [`examples/disk_offloading/kimi_k26_example.py`](https://github.com/vllm-project/llm-compressor/tree/main/examples/disk_offloading/kimi_k26_example.py)

---

## The Converter Protocol

All converters implement the `Converter` protocol defined in `compressed_tensors`. This template shows the current tensor and quantization config contracts. Subclassing `Converter` also inherits its default `update_model_config(model_config)` pass-through for non-quantization fields.

```python
import torch
from compressed_tensors.entrypoints.convert import Converter
from compressed_tensors.quantization import QuantizationConfig

class MyConverter(Converter):
    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Validate meta tensors before conversion. Return tensors with the names,
        shapes, and dtypes that process() will produce for the next converter.
        Override this when process() needs real tensor values.
        """
        return self.process(tensors.copy())

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Transform tensor names and values. Called once per shard.
        May rename, dequantize, repack, or reorder tensors.
        """
        raise NotImplementedError("Implement the source-to-target tensor mapping")

    def update_config(
        self, config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        """
        Receive the existing CT config or the previous converter's output.
        Return an updated/replacement config, or None when dequantizing to dense.
        """
        raise NotImplementedError("Describe the output quantization scheme")

    def get_dependencies(self, weight_name: str) -> set[str]:
        """
        Return any additional weight names that must be loaded alongside
        weight_name to process it correctly (e.g. paired scale tensors that
        live in a different shard).
        """
        return set()
```

`convert_checkpoint` uses `get_dependencies` to build a complete load plan before processing, so converters can safely access cross-shard tensors in `process`.

Validation runs without tensor data on the meta device. Use metadata-only operations, or construct representative output tensors when processing requires values; the inherited `validate` default delegates to `process`. Override `update_model_config` only when the conversion also changes model structure. The pipeline handles loading and saving `config.json`.
