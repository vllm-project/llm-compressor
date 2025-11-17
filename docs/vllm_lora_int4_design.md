# vLLM LoRA + INT4 Integration Design

## Overview

This document describes the design for enabling LoRA adapter injection on INT4 quantized models in vLLM. The approach uses on-demand unpacking of INT4 weights to floating-point format when LoRA adapters are loaded.

## Problem Statement

Currently, vLLM's LoRA injection assumes that model weights are accessible as regular floating-point tensors. However, INT4 quantized models stored in compressed-tensors format use packed buffers (2 INT4 values per byte) that cannot be directly used with LoRA adapters.

### Current State

```python
# vLLM LoRA injection (simplified)
def inject_lora(base_module, lora_adapter):
    # Expects base_module.weight to be FP16/BF16 tensor
    base_weight = base_module.weight  # ❌ Fails: weight is packed uint8
    # Apply LoRA: output = base_weight @ input + lora_A @ lora_B @ input
```

### Desired State

```python
# vLLM LoRA injection with INT4 support
def inject_lora(base_module, lora_adapter):
    # Detect and unpack INT4 weights
    if is_int4_quantized(base_module):
        base_weight = unpack_int4_for_lora(base_module)  # ✅ Unpack to FP16
    else:
        base_weight = base_module.weight
    # Apply LoRA with unpacked weights
```

## Architecture

### Components

1. **Detection Module** (`vllm/model_executor/layers/quantization/compressed_tensors.py`)
   - Detect INT4 quantized models with compressed-tensors format
   - Read LoRA metadata from model config

2. **Unpacking Module** (`vllm/lora/int4_utils.py`)
   - Implement INT4 unpacking logic (can reuse from llm-compressor)
   - Cache unpacked weights to avoid repeated unpacking

3. **LoRA Integration** (`vllm/lora/layers.py`)
   - Modify LoRA injection to handle packed weights
   - Wire LoRA computation into quantized forward pass

4. **Forward Pass** (`vllm/model_executor/models/*.py`)
   - Ensure LoRA operates on FP16 while base model uses INT4 kernels
   - Combine quantized base output with LoRA output

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL LOADING (vLLM)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Load compressed model from disk                            │
│     ├─> Read config.json (contains lora_compatible flag)      │
│     ├─> Read lora_metadata.json                               │
│     └─> Load model.safetensors (packed INT4 weights)          │
│                                                                 │
│  2. Detect INT4 quantization format                            │
│     ├─> Check quantization_config.format == "pack_quantized"  │
│     ├─> Check lora_compatible flag in config                  │
│     └─> Store quantization parameters (scales, zero_points)   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LORA ADAPTER LOADING                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  3. User requests LoRA adapter loading                         │
│     ├─> vllm.load_lora_adapters(["adapter_path"])            │
│     └─> Trigger on-demand unpacking                           │
│                                                                 │
│  4. Unpack INT4 weights (first time only)                     │
│     ├─> For each target module (q_proj, v_proj, etc.):       │
│     │   ├─> Read packed_weight (uint8)                       │
│     │   ├─> Read weight_scale, weight_zero_point            │
│     │   ├─> Unpack: INT4 → FP16                            │
│     │   └─> Cache unpacked weight                           │
│     └─> Store in module as module.weight_lora_base           │
│                                                                 │
│  5. Inject LoRA adapters                                       │
│     ├─> Create LoRA layers (lora_A, lora_B)                  │
│     ├─> Associate with unpacked base weights                  │
│     └─> Register LoRA forward hooks                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE WITH LORA                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  6. Forward pass with INT4 + LoRA                             │
│     ├─> Quantized path (base model):                          │
│     │   └─> quantized_output = int4_kernel(packed_weight, x) │
│     │                                                          │
│     ├─> LoRA path (adapters):                                 │
│     │   └─> lora_output = lora_B @ lora_A @ x               │
│     │                                                          │
│     └─> Combine outputs:                                       │
│         └─> final_output = quantized_output + lora_output    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Detection Logic

**File**: `vllm/model_executor/layers/quantization/compressed_tensors.py`

```python
class CompressedTensorsConfig:
    """Extended to support LoRA detection."""

    def __init__(self, ...):
        # Existing initialization
        self.format = config.get("format")

        # New: LoRA compatibility detection
        self.lora_compatible = config.get("lora_compatible", False)
        self.lora_target_modules = config.get("lora_target_modules", [])

    def is_lora_compatible(self) -> bool:
        """Check if this quantized model supports LoRA."""
        return (
            self.lora_compatible and
            self.format in ["pack_quantized", "marlin_24"]
        )
```

### 2. Unpacking Module

**File**: `vllm/lora/int4_utils.py` (new file)

```python
"""Utilities for unpacking INT4 weights for LoRA compatibility."""

import torch
from typing import Optional, Dict
from loguru import logger


class INT4Unpacker:
    """Manages unpacking and caching of INT4 weights."""

    def __init__(self):
        self._cache: Dict[str, torch.Tensor] = {}

    def unpack_int4_weights(
        self,
        packed_weights: torch.Tensor,
        scales: torch.Tensor,
        zero_points: Optional[torch.Tensor] = None,
        group_size: Optional[int] = None,
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Unpack INT4 quantized weights to floating-point format.

        This is the core unpacking algorithm that converts packed INT4
        weights back to FP16/BF16 for LoRA injection.

        NOTE: This implementation should match the unpacking logic in
        llm-compressor/lora_utils.py for consistency.
        """
        # [Implementation copied from llm-compressor]
        # See: llm-compressor/src/llmcompressor/transformers/compression/lora_utils.py

        # Unpack: extract two INT4 values from each uint8 byte
        out_features, packed_in_features = packed_weights.shape
        in_features = packed_in_features * 2

        unpacked = torch.zeros(
            (out_features, in_features),
            dtype=torch.uint8,
            device=packed_weights.device
        )
        unpacked[:, 0::2] = packed_weights & 0x0F
        unpacked[:, 1::2] = (packed_weights >> 4) & 0x0F

        # Convert to signed and dequantize
        unpacked_signed = unpacked.to(torch.int8) - 8
        unpacked_fp = unpacked_signed.to(output_dtype)

        # Apply zero points and scales
        # [Full implementation in actual code]

        return unpacked_fp

    def unpack_module(
        self,
        module: torch.nn.Module,
        module_name: str,
        force: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        Unpack INT4 weights from a module, with caching.

        Args:
            module: PyTorch module with packed weights
            module_name: Unique name for caching
            force: If True, bypass cache and re-unpack

        Returns:
            Unpacked FP16 weights, or None if not applicable
        """
        # Check cache first
        if not force and module_name in self._cache:
            logger.debug(f"Using cached unpacked weights for {module_name}")
            return self._cache[module_name]

        # Check if module has packed weights
        if not hasattr(module, "weight_packed"):
            return None

        packed_weights = module.weight_packed
        scales = module.weight_scale
        zero_points = getattr(module, "weight_zero_point", None)

        # Infer group size from scales shape
        group_size = None
        if scales.ndim == 2:
            out_features, num_groups = scales.shape
            in_features = packed_weights.shape[1] * 2
            group_size = in_features // num_groups

        # Unpack
        unpacked = self.unpack_int4_weights(
            packed_weights=packed_weights,
            scales=scales,
            zero_points=zero_points,
            group_size=group_size,
        )

        # Cache for future use
        self._cache[module_name] = unpacked
        logger.info(f"Unpacked and cached weights for {module_name}: {unpacked.shape}")

        return unpacked

    def clear_cache(self):
        """Clear the unpacked weights cache to free memory."""
        self._cache.clear()


# Global unpacker instance
_global_unpacker = INT4Unpacker()


def get_unpacker() -> INT4Unpacker:
    """Get the global INT4 unpacker instance."""
    return _global_unpacker
```

### 3. LoRA Layer Modifications

**File**: `vllm/lora/layers.py`

```python
class LoRALayer(torch.nn.Module):
    """Extended to support INT4 base weights."""

    def __init__(
        self,
        base_layer: torch.nn.Module,
        lora_config: LoRAConfig,
        ...
    ):
        super().__init__()
        self.base_layer = base_layer
        self.lora_config = lora_config

        # NEW: Check if base layer is INT4 quantized
        self.is_int4_quantized = hasattr(base_layer, "weight_packed")

        # NEW: Unpack INT4 weights if needed for LoRA
        if self.is_int4_quantized:
            from vllm.lora.int4_utils import get_unpacker
            unpacker = get_unpacker()
            self.base_weight_fp = unpacker.unpack_module(
                base_layer,
                module_name=f"{lora_config.model_name}.{base_layer._name}",
            )
            logger.info(f"Unpacked INT4 weights for LoRA: {self.base_weight_fp.shape}")
        else:
            self.base_weight_fp = base_layer.weight

        # Initialize LoRA adapters (A and B matrices)
        self.lora_A = torch.nn.Parameter(...)
        self.lora_B = torch.nn.Parameter(...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining quantized base and FP LoRA."""

        if self.is_int4_quantized:
            # Use quantized kernel for base model (fast path)
            base_output = self.base_layer.quant_forward(x)

            # Compute LoRA output using unpacked FP weights
            # output = base_output + lora_B @ lora_A @ x
            lora_output = self._lora_forward(x)

            result = base_output + lora_output
        else:
            # Standard LoRA forward for non-quantized models
            result = self.base_layer(x) + self._lora_forward(x)

        return result

    def _lora_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA adapter output."""
        # Standard LoRA computation
        return (x @ self.lora_A.T) @ self.lora_B.T
```

### 4. Model-Specific Changes

**File**: `vllm/model_executor/models/llama.py` (example)

```python
class LlamaAttention(torch.nn.Module):
    """Extended to support LoRA with INT4."""

    def __init__(...):
        # Existing initialization
        self.q_proj = LinearMethod(...)
        self.k_proj = LinearMethod(...)
        self.v_proj = LinearMethod(...)

        # NEW: Check if LoRA adapters should be prepared
        if self.lora_config and self._is_quantized():
            self._prepare_lora_adapters()

    def _is_quantized(self) -> bool:
        """Check if this layer is quantized."""
        return hasattr(self.q_proj, "weight_packed")

    def _prepare_lora_adapters(self):
        """Prepare unpacked weights for LoRA injection."""
        from vllm.lora.int4_utils import get_unpacker
        unpacker = get_unpacker()

        # Unpack weights for common LoRA targets
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            module = getattr(self, name)
            unpacker.unpack_module(module, module_name=f"attention.{name}")
```

## Memory Considerations

### Memory Overhead

- **Packed INT4**: 0.5 bytes per parameter
- **Unpacked FP16**: 2 bytes per parameter
- **Overhead**: 4x increase when weights are unpacked

### Optimization Strategies

1. **Selective Unpacking**: Only unpack LoRA target modules (q_proj, v_proj, etc.)
   - Typical: ~25% of model parameters
   - Example: 7B model = 1.75GB unpacked instead of 7GB

2. **Lazy Unpacking**: Unpack on-demand when LoRA is first loaded
   - Zero overhead if LoRA is not used
   - Amortize cost across many inference requests

3. **Cache Management**: Keep unpacked weights in memory while LoRA is active
   - Clear cache when LoRA adapters are unloaded
   - Trade-off: memory vs. re-unpacking cost

### Example Memory Usage

For a Llama-2-7B model with INT4 quantization + LoRA:

| Component | Memory |
|-----------|--------|
| Base model (INT4 packed) | ~3.5 GB |
| Unpacked LoRA targets (FP16) | ~1.75 GB |
| LoRA adapters (r=16) | ~50 MB |
| **Total** | **~5.25 GB** |

Compare to:
- FP16 model: ~14 GB
- INT4 without LoRA: ~3.5 GB

## Testing Strategy

### Unit Tests

1. **Test INT4 unpacking correctness**
   ```python
   def test_unpack_int4_weights():
       # Create known packed weights
       # Verify unpacking produces expected FP16 values
       pass
   ```

2. **Test LoRA injection with INT4**
   ```python
   def test_lora_injection_int4():
       # Load INT4 quantized model
       # Inject LoRA adapters
       # Verify forward pass works
       pass
   ```

3. **Test memory overhead**
   ```python
   def test_memory_overhead():
       # Measure memory before/after unpacking
       # Verify cache clearing works
       pass
   ```

### Integration Tests

1. **End-to-end inference test**
   - Quantize model with llm-compressor (INT4)
   - Load in vLLM
   - Inject LoRA adapters
   - Run inference and verify outputs

2. **Performance benchmarks**
   - Measure latency: INT4 vs INT4+LoRA
   - Measure throughput: tokens/second
   - Compare to FP16 baseline

### Validation

1. **Accuracy tests**
   - Compare outputs: INT4+LoRA vs FP16+LoRA
   - Ensure unpacking doesn't introduce numerical errors
   - Verify LoRA adapters work correctly

2. **Compatibility tests**
   - Test with different quantization configs (grouped, channel, etc.)
   - Test with different LoRA ranks (r=8, 16, 32)
   - Test with multiple LoRA adapters

## Implementation Checklist

### Phase 1: Core Infrastructure (llm-compressor) ✅

- [x] Create INT4 unpacking utilities
- [x] Add LoRA metadata to compressed models
- [x] Update compression pipeline to save metadata
- [x] Add tests for unpacking utilities

### Phase 2: vLLM Integration (POC)

- [ ] Add INT4 unpacking module to vLLM
- [ ] Extend quantization config for LoRA detection
- [ ] Modify LoRA layers to handle packed weights
- [ ] Implement on-demand unpacking with caching
- [ ] Update model classes (Llama, Mistral, etc.)

### Phase 3: Testing & Validation

- [ ] Create end-to-end test in llm-compressor
- [ ] Add unit tests in vLLM
- [ ] Run integration tests
- [ ] Benchmark performance and memory

### Phase 4: Documentation & Contribution

- [ ] Write user-facing documentation
- [ ] Create example notebooks
- [ ] Prepare vLLM PR with detailed description
- [ ] Address review feedback

## Usage Example

### Quantize Model (llm-compressor)

```python
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Quantize to INT4
oneshot(
    model=model,
    dataset="c4",
    recipe="int4_awq_recipe.yaml",  # INT4 with group quantization
    output_dir="./llama2-7b-int4",
)

# Model is saved with LoRA metadata
# - config.json includes lora_compatible: true
# - lora_metadata.json contains unpacking info
```

### Load in vLLM with LoRA

```python
from vllm import LLM

# Load INT4 quantized model
llm = LLM(
    model="./llama2-7b-int4",
    quantization="compressed-tensors",  # Auto-detected
)

# Load LoRA adapters (triggers on-demand unpacking)
llm.load_lora_adapters([
    {"name": "adapter1", "path": "./lora_adapters/adapter1"},
])

# Run inference with LoRA
outputs = llm.generate(
    "Hello, how are you?",
    lora_request={"lora_name": "adapter1"},
)

print(outputs[0].text)
```

## Future Enhancements

1. **INT4 LoRA Adapters**: Quantize LoRA weights to INT4 for further memory savings

2. **Fused Kernels**: Develop custom CUDA kernels that combine INT4 matmul + LoRA

3. **Multi-LoRA Batching**: Support multiple LoRA adapters in the same batch

4. **Dynamic Quantization**: Allow switching between packed/unpacked modes

## References

- [llm-compressor documentation](https://github.com/vllm-project/llm-compressor)
- [vLLM LoRA documentation](https://docs.vllm.ai/en/latest/models/lora.html)
- [compressed-tensors format specification](https://github.com/neuralmagic/compressed-tensors)
- [AWQ quantization paper](https://arxiv.org/abs/2306.00978)
