# HIGGS: Heuristic ILP-Guided Grouped Scheme Mixed-Precision Quantization

HIGGS is an automated mixed-precision quantization system that uses Integer Linear Programming (ILP) to select optimal quantization schemes for each layer in a neural network.

## Overview

Traditional uniform quantization applies the same bitwidth to all layers. HIGGS automatically determines which layers can tolerate aggressive quantization (e.g., 4-bit) and which need higher precision (e.g., 8-bit), minimizing quality degradation for a target compression ratio.

**Key Features:**
- **Model-free**: Works directly on safetensors files, no model definition needed
- **ILP-based optimization**: Globally optimal scheme assignment
- **Heuristic layer importance**: Weighs layers by size, depth, and type
- **Fusion-aware**: Respects hardware constraints (fused layers get same scheme)
- **Flexible**: Supports any set of candidate quantization schemes

## Quick Start

```python
from llmcompressor.transformers.compression.higgs import ilp_quantize

# Quantize a model with automatic mixed-precision selection
config = ilp_quantize(
    model_stub="meta-llama/Llama-3-8B",
    save_directory="./Llama-3-8B-HIGGS",
    candidate_schemes=["W4A16", "W8A8", "FP8_DYNAMIC"],
    targets="Linear",
    ignore=["lm_head"],
    max_workers=8,
    device="cuda:0",
)

print(f"Generated {len(config.config_groups)} config groups")
```

## How It Works

### Phase 1: MSE Collection & ILP Optimization (No Save)

1. **For each layer and each candidate scheme**:
   - Quantize the layer's weights
   - Compute Mean Squared Error (MSE) vs. original weights
   - Discard quantized weights

2. **Compute layer importance (alpha)**:
   ```
   alpha = log(num_params + 1) × (1 + depth × 0.05) × type_multiplier
   ```
   Where `type_multiplier` is:
   - Embedding: 1.5
   - Attention: 1.2
   - MLP: 0.9
   - Default: 1.0

3. **Solve ILP**:
   ```
   Minimize: Σ (MSE[layer][scheme] × alpha[layer] × x[layer][scheme])
   
   Subject to:
   - Each layer gets exactly one scheme: Σ x[layer][scheme] = 1
   - Fused layers use same scheme: x[layer_i][s] = x[layer_j][s]
   - Optional: Average bitwidth ≤ target
   ```

### Phase 2: Apply Quantization (With Save)

- Use the ILP-selected scheme for each layer
- Quantize using llmcompressor's quantization infrastructure
- Save the quantized model to disk

## Architecture

```
src/llmcompressor/transformers/compression/higgs/
├── __init__.py                  # ilp_quantize() API
├── mse_utils.py                 # MSE computation
├── alpha_heuristic.py           # Layer importance calculation
├── fusion_utils.py              # Fused layer detection
├── ilp_solver.py                # ILP formulation and solving
├── config_generator.py          # QuantizationConfig generation
├── mse_collector.py             # Phase 1 converter
└── quantization_converter.py    # Phase 2 converter
```

## Advanced Usage

### Custom Candidate Schemes

```python
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

schemes = [
    QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, type="int", strategy="channel"),
    ),
    QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=8, type="int", strategy="tensor"),
    ),
]

config = ilp_quantize(
    model_stub="...",
    save_directory="...",
    candidate_schemes=schemes,
)
```

### Constrain Average Bitwidth

```python
config = ilp_quantize(
    model_stub="...",
    save_directory="...",
    candidate_schemes=["W4A16", "W8A8"],
    target_avg_bitwidth=4.5,  # Average 4.5 bits per weight
)
```

### Manual Two-Phase Process

```python
from llmcompressor.transformers.compression.higgs import (
    HiggsMSECollectorConverter,
    HiggsQuantizationConverter,
    compute_heuristic_alphas,
    detect_fused_groups,
)
from compressed_tensors.entrypoints.convert import convert_checkpoint

# Phase 1: Collect MSE and solve ILP
collector = HiggsMSECollectorConverter(
    candidate_schemes=["W4A16", "W8A8"],
    targets="Linear",
    alpha_calculator=compute_heuristic_alphas,
    fusion_detector=detect_fused_groups,
)

# Process shards manually...
# (see implementation in ilp_quantize() for details)

optimal_config = collector.create_config()

# Phase 2: Apply quantization
quantizer = HiggsQuantizationConverter(
    optimal_config=optimal_config,
    targets="Linear",
)

convert_checkpoint(
    model_stub="...",
    save_directory="...",
    converter=quantizer,
)
```

## Testing

All components have comprehensive unit tests:

```bash
pytest tests/llmcompressor/transformers/compression/higgs/ -v
```

## References

- ILP solver: PuLP with CBC backend
- Quantization: llmcompressor + compressed-tensors
