## DeepSeek V3.2 NVFP4-FP8-BLOCK Example

### Code Walkthrough

Let's walk through the main steps of the quantization process:
1. Convert checkpoint to bfloat16
2. Merge config files
3. Load model with disk offloading
4. Configure quantization algorithm and scheme
5. Apply quantization
6. Save to disk in compressed-tensors format

### 1. Convert Checkpoint to bfloat16

The original `deepseek-ai/DeepSeek-V3.2` checkpoint has layers quantized in the FP8_BLOCK scheme. We first convert it to bfloat16 so that we can compress it in compressed-tensors format with a different configuration.

```python
from compressed_tensors.entrypoints.convert import (
    FP8BlockDequantizer,
    convert_checkpoint,
)

MODEL_ID = "deepseek-ai/DeepSeek-V3.2"
BFLOAT16_SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-bf16"

convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=BFLOAT16_SAVE_DIR,
    converter=FP8BlockDequantizer(
        # `deepseek-ai/DeepSeek-V3.2` fp8-block-quantized layers, found by inspection
        targets=[
            r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
            r"re:.*self_attn.*\.(kv_b|o|q_a|q_b)_proj$",
            r"re:.*self_attn.kv_a_proj_with_mqa$",
            r"re:.*self_attn.indexer.(wk|wq_b)$",
        ],
    ),
    max_workers=4,
)
```

### 2. Merge Config Files

DeepSeek splits important config info into a separate file that will break loading in transformers if not merged into `config.json`.

```python
import json

with open(f"{BFLOAT16_SAVE_DIR}/config.json", "r") as f:
    orig_config = json.load(f)
with open(f"{BFLOAT16_SAVE_DIR}/inference/config_671B_v3.2.json", "r") as f:
    additional_config_data = json.load(f)
    additional_config_data.pop("dtype")
with open(f"{BFLOAT16_SAVE_DIR}/config.json", "w") as f:
    config = orig_config | additional_config_data
    json.dump(config, f)
```

### 3. Load Model with Disk Offloading

DeepSeek V3.2 is a very large model. We use disk offloading to fit it in memory by loading as much as possible on CPU, with the rest going to disk.

```python
import torch
from compressed_tensors.offload import load_offloaded_model
from transformers import AutoTokenizer

from llmcompressor.modeling.deepseekv32.model import DeepseekV32ForCausalLM

with load_offloaded_model(), torch.no_grad():
    model = DeepseekV32ForCausalLM.from_pretrained(
        BFLOAT16_SAVE_DIR,
        dtype="auto",
        device_map="auto_offload",  # fit as much as possible on cpu, rest goes on disk
        trust_remote_code=True,
        offload_folder="./offload_folder",
        max_memory={"cpu": 500e9},  # don't exceed 500GB RAM
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
```

### 4. Configure the Quantization Algorithm and Scheme

We configure the quantization to:
- Quantize MLP weights to NVFP4
- Quantize self-attention weights to FP8_BLOCK

This is a calibrated flow (i.e. requiring a calibration dataset).

```python
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Select calibration dataset.
DATASET_ID = "ultrachat-200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 2048

recipe = QuantizationModifier(
    config_groups={
        "config_group_0": QuantizationScheme(
            targets=[
                r"re:model.*mlp.*(gate|up|down|gate_up)_proj$",
            ],
            **NVFP4,
        ),
        "config_group_1": QuantizationScheme(
            targets=[
                # NOTE: leaving weights_proj in bf16
                r"re:model.*self_attn.indexer.(wk|wq_b)$",
                r"re:model.*self_attn.kv_a_proj_with_mqa$",
                r"re:model.*self_attn.(kv_b|o|q_a|q_b)_proj$",
            ],
            **FP8_BLOCK,
        ),
    },
    ignore=["lm_head"],
)
```

### 5. Apply Quantization

```python
oneshot(
    model=model,
    processor=tokenizer,
    dataset=DATASET_ID,
    splits={"calibration": f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"},
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
```

### 6. Save to Disk in Compressed-Tensors Format

```python
SAVE_DIR = "DeepSeek-V3.2-NVFP4-FP8-BLOCK"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

---

**Full example script:** [`examples/disk_offloading/deepseek_v32_example.py`](https://github.com/vllm-project/llm-compressor/blob/main/examples/disk_offloading/deepseek_v32_example.py)

**Quantized checkpoint:** [RedHatAI/DeepSeek-V3.2-NVFP4-FP8-BLOCK](https://huggingface.co/RedHatAI/DeepSeek-V3.2-NVFP4-FP8-BLOCK)
