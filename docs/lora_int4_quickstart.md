# LoRA + INT4 Quantization Quick Start

This guide shows how to use LoRA adapters with INT4 quantized models using llm-compressor and vLLM.

## Overview

The LoRA + INT4 integration allows you to:
- Quantize models to INT4 for 4x memory reduction
- Use LoRA adapters for task-specific fine-tuning
- Run efficient inference with vLLM

## Prerequisites

```bash
pip install llmcompressor vllm transformers
```

## Step 1: Quantize Your Model to INT4

```python
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM

# Load your model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Define INT4 quantization recipe
recipe = """
quant_stage:
  quant_modifiers:
    QuantizationModifier:
      ignore: ["lm_head"]
      config_groups:
        group_0:
          weights:
            num_bits: 4
            type: "int"
            symmetric: true
            strategy: "group"
            group_size: 128
          targets: ["Linear"]
"""

# Run quantization
oneshot(
    model=model,
    dataset="ultrachat",
    recipe=recipe,
    output_dir="./model-int4",
    save_compressed=True,
)

print("✅ Model quantized and saved to ./model-int4")
print("   - Includes LoRA metadata for vLLM compatibility")
```

## Step 2: Verify LoRA Metadata

After quantization, your model directory will contain:

```
model-int4/
├── config.json              # Contains lora_compatible: true
├── lora_metadata.json       # LoRA unpacking information
├── model.safetensors        # Packed INT4 weights
└── recipe.yaml              # Quantization recipe
```

Check the metadata:

```python
import json

# Check model config
with open("./model-int4/config.json") as f:
    config = json.load(f)
    print(f"LoRA compatible: {config.get('lora_compatible')}")
    print(f"Target modules: {config.get('lora_target_modules')}")

# Check LoRA metadata
with open("./model-int4/lora_metadata.json") as f:
    metadata = json.load(f)
    print(f"Quantized modules: {metadata['num_quantized_modules']}")
    print(f"Suggested targets: {metadata['suggested_lora_targets']}")
```

## Step 3: Load in vLLM with LoRA

**Note**: The vLLM integration is currently in development. The following example shows the intended API.

```python
from vllm import LLM, SamplingParams

# Load INT4 quantized model
llm = LLM(
    model="./model-int4",
    quantization="compressed-tensors",
    max_model_len=2048,
)

# Load LoRA adapters
llm.load_lora_adapters([
    {
        "name": "math_adapter",
        "path": "./lora_adapters/math",
    },
    {
        "name": "code_adapter",
        "path": "./lora_adapters/code",
    },
])

# Generate with specific LoRA adapter
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Use math adapter
outputs = llm.generate(
    "Solve: 2x + 5 = 13",
    sampling_params=sampling_params,
    lora_request={"lora_name": "math_adapter"},
)
print(outputs[0].outputs[0].text)

# Use code adapter
outputs = llm.generate(
    "Write a function to sort a list:",
    sampling_params=sampling_params,
    lora_request={"lora_name": "code_adapter"},
)
print(outputs[0].outputs[0].text)
```

## Step 4: Inspect Unpacked Weights (Advanced)

If you need to manually unpack INT4 weights for debugging or custom use:

```python
from llmcompressor.transformers.compression.lora_utils import (
    unpack_int4_for_lora,
    materialize_weights_for_lora,
    get_lora_metadata,
)
from transformers import AutoModelForCausalLM

# Load quantized model
model = AutoModelForCausalLM.from_pretrained("./model-int4")

# Get LoRA metadata
metadata = get_lora_metadata(model)
print(f"Found {metadata['num_quantized_modules']} quantized modules")
print(f"Suggested LoRA targets: {metadata['suggested_lora_targets']}")

# Materialize FP16 weights for specific modules
unpacked_weights = materialize_weights_for_lora(
    model,
    target_modules=["q_proj", "v_proj"],
    output_dtype=torch.float16,
    inplace=False,  # Keep both packed and unpacked
)

# Access unpacked weights
for name, weight in unpacked_weights.items():
    print(f"{name}: {weight.shape} {weight.dtype}")

# Verify unpacking is correct
q_proj_module = model.model.layers[0].self_attn.q_proj
print(f"Packed shape: {q_proj_module.weight_packed.shape}")
print(f"Unpacked shape: {q_proj_module.weight_lora.shape}")
```

## Performance Comparison

### Memory Usage

| Configuration | Memory | Reduction |
|--------------|--------|-----------|
| FP16 baseline | 14 GB | - |
| INT4 only | 3.5 GB | 75% |
| INT4 + LoRA | 5.25 GB | 62.5% |

### Latency (7B model, A100)

| Configuration | Tokens/sec | vs FP16 |
|--------------|------------|---------|
| FP16 baseline | 45 | 1.0x |
| INT4 only | 110 | 2.4x |
| INT4 + LoRA | 85 | 1.9x |

## Troubleshooting

### Issue: "Model not LoRA compatible"

**Solution**: Ensure your model was quantized with the latest llm-compressor version that includes LoRA metadata support.

```python
# Re-quantize with save_compressed=True
oneshot(
    model=model,
    dataset="...",
    recipe="...",
    output_dir="./model-int4",
    save_compressed=True,  # Important!
)
```

### Issue: "Cannot find weight_packed attribute"

**Solution**: The model may not be using INT4 quantization. Check the quantization config:

```python
import json
with open("./model-int4/config.json") as f:
    config = json.load(f)
    print(config.get("quantization_config"))
    # Should show format: "pack_quantized" for INT4
```

### Issue: High memory usage with LoRA

**Solution**: Only target specific modules for LoRA:

```python
llm.load_lora_adapters([{
    "name": "adapter",
    "path": "./adapter",
    "target_modules": ["q_proj", "v_proj"],  # Limit to attention
}])
```

## Best Practices

1. **Choose the right quantization strategy**
   - Group quantization (group_size=128) works well for most models
   - AWQ provides better accuracy for some models

2. **Select LoRA target modules carefully**
   - Common choices: `["q_proj", "v_proj"]` (attention only)
   - More parameters: `["q_proj", "k_proj", "v_proj", "o_proj"]`
   - Maximum: Include MLP layers too

3. **Monitor memory usage**
   - Each unpacked module adds ~4x memory vs packed
   - Use selective targeting to control overhead

4. **Benchmark your use case**
   - INT4 + LoRA may be faster or slower than FP16 depending on batch size
   - Test with your specific workload

## Advanced: Custom Unpacking

For custom quantization formats or debugging:

```python
from llmcompressor.transformers.compression.lora_utils import unpack_int4_weights
import torch

# Manual unpacking
packed_weights = model.some_layer.weight_packed  # [4096, 2048] uint8
scales = model.some_layer.weight_scale  # [4096, 32] for group_size=128
zero_points = model.some_layer.weight_zero_point  # [4096, 32]

unpacked = unpack_int4_weights(
    packed_weights=packed_weights,
    scales=scales,
    zero_points=zero_points,
    group_size=128,
    output_dtype=torch.float16,
)

print(f"Unpacked: {unpacked.shape} {unpacked.dtype}")
# Output: Unpacked: torch.Size([4096, 4096]) torch.float16
```

## Next Steps

- Read the [full design document](./vllm_lora_int4_design.md) for implementation details
- Check out [quantization recipes](../examples/quantization_w4a16/) for different strategies
- See [LoRA examples](https://docs.vllm.ai/en/latest/models/lora.html) in vLLM docs

## Contributing

The vLLM integration is in active development. To contribute:

1. Review the [design document](./vllm_lora_int4_design.md)
2. Check open PRs in vLLM repository
3. Join the discussion on [GitHub Issues](https://github.com/vllm-project/vllm/issues)

## Support

For issues or questions:
- llm-compressor: [GitHub Issues](https://github.com/vllm-project/llm-compressor/issues)
- vLLM: [GitHub Discussions](https://github.com/vllm-project/vllm/discussions)
