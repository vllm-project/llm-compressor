## Qwen3.8-27B 4-bit Quantization Example

Qwen3.8-27B is a vision-language model. This example produces 4-bit checkpoints with AWQ followed by GPTQ, covering three formats:

- **W4A16** (INT4): weight-only 4-bit quantization with 16-bit activations
- **NVFP4**: 4-bit weights and activations with per-group-16 scaling
- **MXFP4**: 4-bit weights and activations with a shared E8M0 scale

The full example scripts can be found here:

- [W4A16 example](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a16/qwen3_8_gptq_awq_example.py)
- [NVFP4 example](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a4_fp4/qwen3_8_gptq_awq_example.py)
- [MXFP4 example](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a4_mxfp4/qwen3_8_gptq_awq_example.py)

### Code Walkthrough

Let's walk through the main steps of the quantization process:
1. Load model
2. Configure quantization algorithm and scheme
3. Prepare calibration dataset
4. Apply quantization
5. Save to disk in compressed-tensors format

### 1. Load Model

```python
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen3.8-27B"

# Qwen3.8-27B is a vision-language model, so it must be loaded with its
# Qwen3_5ForConditionalGeneration class to keep the vision tower.
with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)
```

### 2. Configure Quantization Algorithm and Scheme

All three formats use AWQ to smooth activations before GPTQ performs the weight quantization. The vision tower, `lm_head`, embeddings, and the Gated DeltaNet projections that are incompatible with 4-bit quantization are skipped.

#### W4A16 (INT4)

```python
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

recipe = [
    AWQModifier(duo_scaling="both"),
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
            "re:visual.*",
            "re:model.visual.*",
            r"re:.*lm_head",
            "re:.*embed_tokens$",
            r"re:.*linear_attn\.in_proj_a$",
            r"re:.*linear_attn\.in_proj_b$",
        ],
    ),
]
```

#### NVFP4

For NVFP4, attention projections, `lm_head`, and the MLP layers in the last eight decoder blocks are kept in FP8 to preserve accuracy, while the remaining MLP projections are quantized to NVFP4. The KV cache is quantized to FP8 as well.

```python
from compressed_tensors.quantization.quant_scheme import (
    FP8_DYNAMIC,
    NVFP4,
    QuantizationScheme,
)
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

recipe = [
    AWQModifier(duo_scaling="both"),
    GPTQModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=[
                    r"re:.*self_attn\.(q|k|v|o)_proj$",
                    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
                    r"re:.*lm_head",
                    r"re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\..*(gate|up|down)_proj$",
                ],
                **FP8_DYNAMIC,
            ),
            "mlp": QuantizationScheme(
                targets=[r"re:.*mlp\..*(gate|up|down)_proj$"],
                **NVFP4,
            ),
        },
        ignore=[
            "re:visual.*",
            "re:model.visual.*",
        ],
        kv_cache_scheme={
            "num_bits": 8,
            "type": "float",
            "symmetric": True,
            "strategy": "tensor",
            "dynamic": False,
            "observer": "static_minmax",
        },
    ),
]
```

#### MXFP4

```python
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

recipe = [
    AWQModifier(duo_scaling="both"),
    GPTQModifier(
        targets="Linear",
        scheme="MXFP4",
        ignore=[
            "re:visual.*",
            "re:model.visual.*",
            r"re:.*lm_head",
            "re:.*embed_tokens$",
            r"re:.*linear_attn\.in_proj_a$",
            r"re:.*linear_attn\.in_proj_b$",
        ],
        kv_cache_scheme={
            "num_bits": 8,
            "type": "float",
            "symmetric": True,
            "strategy": "tensor",
            "dynamic": False,
            "observer": "static_minmax",
        },
    ),
]
```

### 3. Prepare Calibration Dataset

The examples use the prebaked `perfectblend` calibration dataset with 512 samples and a maximum sequence length of 4096 tokens.

### 4. Apply Quantization

```python
from llmcompressor import oneshot

oneshot(
    model=model,
    processor=processor,
    recipe=recipe,
    dataset="perfectblend",
    splits="train[:512]",
    max_seq_length=4096,
    num_calibration_samples=512,
)
```

### 5. Save to Disk in Compressed-Tensors Format

```python
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-GPTQ-AWQ"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```

Use the corresponding suffix for the other formats: `-NVFP4-GPTQ-AWQ` for NVFP4 and `-MXFP4` for MXFP4.

### Calibration-free MXFP6A8

A calibration-free alternative is also available. It keeps the dense MLP projection weights in MXFP6, quantizes attention, Gated DeltaNet, and MTP projections to MXFP8, and keeps `lm_head` in FP8 with per-channel weights. The full script is [here](https://github.com/vllm-project/llm-compressor/blob/main/examples/model_free_ptq/qwen3_8_27b_mxfp6.py).

```python
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.quantization.quant_scheme import (
    FP8_DYNAMIC,
    MXFP6A8,
    MXFP8,
)

from llmcompressor import model_free_ptq

config = QuantizationConfig(
    config_groups={
        "mtp": QuantizationScheme(targets=[r"re:.*mtp\.layers\..*"], **MXFP8),
        "mlp": QuantizationScheme(
            targets=[r"re:.*mlp\.(gate|up|down)_proj$"], **MXFP6A8
        ),
        "attn": QuantizationScheme(
            targets=[
                r"re:.*self_attn\.[qkvo]_proj$",
                r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
            ],
            **MXFP8,
        ),
        "lm_head": QuantizationScheme(targets=[r"re:.*lm_head$"], **FP8_DYNAMIC),
    }
)

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP6A8",
    config=config,
    ignore=[
        r"re:.*visual\..*",
        r"re:.*in_proj_(a|b|ba)$",
        r"re:.*mtp\.fc$",
        r"re:.*embed_tokens$",
    ],
    device="cuda:0",
)
```
