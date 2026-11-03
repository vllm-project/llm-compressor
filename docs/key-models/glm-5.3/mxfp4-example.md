## GLM-5.3 MXFP4 Example

The original [GLM-5.3 checkpoint](https://huggingface.co/zai-org/GLM-5.3) ships in FP8-block format. This example dequantizes it back to full precision and requantizes the model to MXFP4 using model-free PTQ.

The full example script can be found [here](https://github.com/vllm-project/llm-compressor/blob/main/examples/model_free_ptq/glm_5_3_mxfp4.py).

### Code Walkthrough

1. Set the model and ignore list
2. Dequantize and quantize with `model_free_ptq`
3. Save to disk

### 1. Configure the Quantization

`FP8BlockDequantizer` reads the source FP8-block checkpoint and dequantizes it on the fly. Modules that were not quantized in the source checkpoint are listed in `ignore` so that the dequantizer and the MXFP4 quantization step skip them consistently.

```python
from compressed_tensors.entrypoints.convert import FP8BlockDequantizer

MODEL_ID = "zai-org/GLM-5.3"

# Modules that were not FP8-block quantized in the source checkpoint
ignore = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
    "re:.*embed_tokens$",
    "re:.*eh_proj$",
    "re:.*self_attn.indexer.weights_proj$",
]
```

### 2. Dequantize and Quantize

`wk` is dequantized from FP8 by the converter, but vLLM fuses `wk` and `weights_proj` into a dense layer whose FP8-only load path cannot unpack MXFP4 weights. It is therefore kept in bf16.

```python
from llmcompressor import model_free_ptq

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4"

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="MXFP4",
    # wk IS fp8 in the source (dequantizer dequantizes it), but vLLM fuses
    # wk + weights_proj into a dense (quant_config=None) layer whose fp8-only
    # load path can't unpack mxfp4 weights — keep wk in bf16 instead
    ignore=ignore + ["re:.*self_attn.indexer.wk$"],
    converter=FP8BlockDequantizer(ignore=ignore),
    device="cuda:0",
)
```

### 3. Save to Disk

`model_free_ptq` writes the checkpoint to `SAVE_DIR` in compressed-tensors format.
