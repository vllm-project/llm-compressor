## GLM-5.3 NVFP4 Layerwise Example

This example requantizes the published [`RedHatAI/GLM-5.3-MXFP4`](https://huggingface.co/RedHatAI/GLM-5.3-MXFP4) checkpoint to NVFP4 using layerwise decompression and compression. Processing one layer at a time keeps memory usage bounded for this large MoE model.

The full example script can be found [here](https://github.com/vllm-project/llm-compressor/blob/main/examples/layerwise_decompression/glm_5_3_layerwise_example.py).

### Code Walkthrough

1. Load the compressed checkpoint
2. Configure the NVFP4 recipe
3. Apply layerwise decompression and compression
4. Save to disk

### 1. Load the Compressed Checkpoint

The source checkpoint is already compressed to MXFP4, so extend its existing `ignore` list before loading. `device_map="auto_offload"` streams weights from CPU as they are needed.

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor.utils import load_context

MODEL_ID = "RedHatAI/GLM-5.3-MXFP4"

IGNORE = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
    "re:.*embed_tokens$",
    "re:.*eh_proj$",
    "re:.*self_attn.indexer.weights_proj$",
]

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config.quantization_config["ignore"] = [
    *config.quantization_config.get("ignore", []),
    *IGNORE,
]

with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        device_map="auto_offload",
        offload_folder="./offload_folder",
        dtype="auto",
        trust_remote_code=True,
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### 2. Configure the NVFP4 Recipe

```python
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=IGNORE,
)
```

### 3. Apply Layerwise Decompression and Compression

`layerwise_decompression=True` decompresses each layer immediately before it is quantized, and `layerwise_compression=True` compresses it again as soon as quantization finishes. The sequential targets keep the model's attention and MoE blocks as the atomic units.

```python
from llmcompressor import oneshot

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-Layerwise"

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset="perfectblend",
    splits="train[:512]",
    batch_size=4,
    recipe=recipe,
    trust_remote_code_model=True,
    max_seq_length=2048,
    num_calibration_samples=512,
    shuffle_calibration_samples=False,
    pipeline="sequential",
    sequential_targets=["GlmMoeDsaAttention", "GlmMoeDsaMoE"],
    layerwise_decompression=True,
    layerwise_compression=True,
)
```

### 4. Save to Disk

```python
model.generation_config.top_p = None
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```
