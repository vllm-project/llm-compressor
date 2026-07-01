# SafeTensors Key Name Conversions

This document describes the key name transformations applied by `convert_safetensors_keys.py`.

## Transformation Rules

### Top-Level Keys
| Original (Quantized) | Converted (BF16 Format) | Note |
|---------------------|------------------------|------|
| `head.weight` | `lm_head.weight` | LM head weight |
| `model.embed_tokens.weight` | `embed.weight` | Embedding layer |
| `model.norm.*` | `norm.*` | Model normalization |
| `model.layers.X.*` | `layers.X.*` | Strip `model.` prefix |

### Layer Normalization
| Original (Quantized) | Converted (BF16 Format) | Note |
|---------------------|------------------------|------|
| `layers.X.input_layernorm.weight` | `layers.X.attn_norm.weight` | Attention pre-norm |
| `layers.X.post_attention_layernorm.weight` | `layers.X.ffn_norm.weight` | FFN pre-norm |
| `layers.X.self_attn.*` | `layers.X.attn.*` | Self-attention → attn |
| `layers.X.attn.q_a_norm.weight` | `layers.X.attn.q_norm.weight` | Query normalization |
| `layers.X.attn.compressor.kv_norm.weight` | `layers.X.attn.compressor.norm.weight` | Compressor norm |
| `layers.X.attn.compressor.indexer.kv_norm.weight` | `layers.X.attn.indexer.compressor.norm.weight` | Indexer compressor norm |

### Attention Projections
| Original (Quantized) | Converted (BF16 Format) | Note |
|---------------------|------------------------|------|
| `layers.X.attn.sinks` | `layers.X.attn.attn_sink` | Attention sink |
| `layers.X.attn.kv_proj.*` | `layers.X.attn.wkv.*` | KV projection |
| `layers.X.attn.o_a_proj.*` | `layers.X.attn.wo_a.*` | Output A projection |
| `layers.X.attn.o_b_proj.*` | `layers.X.attn.wo_b.*` | Output B projection |
| `layers.X.attn.q_a_proj.*` | `layers.X.attn.wq_a.*` | Query A projection |
| `layers.X.attn.q_b_proj.*` | `layers.X.attn.wq_b.*` | Query B projection |

### Compressor Projections
| Original (Quantized) | Converted (BF16 Format) | Note |
|---------------------|------------------------|------|
| `layers.X.attn.compressor.gate_proj.*` | `layers.X.attn.compressor.wgate.*` | Gate projection |
| `layers.X.attn.compressor.kv_proj.*` | `layers.X.attn.compressor.wkv.*` | KV projection |
| `layers.X.attn.compressor.position_bias` | `layers.X.attn.compressor.ape` | Positional encoding |

### Indexer Compressor Projections
| Original (Quantized) | Converted (BF16 Format) | Note |
|---------------------|------------------------|------|
| `layers.X.attn.compressor.indexer.gate_proj.*` | `layers.X.attn.indexer.compressor.wgate.*` | Gate (reordered) |
| `layers.X.attn.compressor.indexer.kv_proj.*` | `layers.X.attn.indexer.compressor.wkv.*` | KV (reordered) |
| `layers.X.attn.compressor.indexer.position_bias` | `layers.X.attn.indexer.compressor.ape` | APE (reordered) |
| `layers.X.attn.compressor.indexer.q_b_proj.*` | `layers.X.attn.indexer.wq_b.*` | Query B (reordered) |
| `layers.X.attn.compressor.indexer.scorer.weights_proj.*` | `layers.X.attn.indexer.weights_proj.*` | Weights (flattened) |

### FFN/MLP Transformations
| Original (Quantized) | Converted (BF16 Format) | Note |
|---------------------|------------------------|------|
| `layers.X.mlp.*` | `layers.X.ffn.*` | MLP → FFN |
| `layers.X.ffn.experts.Y.gate_proj.*` | `layers.X.ffn.experts.Y.w1.*` | Expert gate projection |
| `layers.X.ffn.experts.Y.up_proj.*` | `layers.X.ffn.experts.Y.w3.*` | Expert up projection |
| `layers.X.ffn.experts.Y.down_proj.*` | `layers.X.ffn.experts.Y.w2.*` | Expert down projection |
| `layers.X.ffn.shared_experts.gate_proj.*` | `layers.X.ffn.shared_experts.w1.*` | Shared gate projection |
| `layers.X.ffn.shared_experts.up_proj.*` | `layers.X.ffn.shared_experts.w3.*` | Shared up projection |
| `layers.X.ffn.shared_experts.down_proj.*` | `layers.X.ffn.shared_experts.w2.*` | Shared down projection |

### Hadamard Compression (HC) Keys
| Original (Quantized) | Converted (BF16 Format) | Note |
|---------------------|------------------------|------|
| `model.hc_head.hc_base` | `hc_head_base` | HC head base (flatten) |
| `model.hc_head.hc_fn` | `hc_head_fn` | HC head function |
| `model.hc_head.hc_scale` | `hc_head_scale` | HC head scale |
| `layers.X.attn_hc.base` | `layers.X.hc_attn_base` | Attn HC base (reorder) |
| `layers.X.attn_hc.fn` | `layers.X.hc_attn_fn` | Attn HC function |
| `layers.X.attn_hc.scale` | `layers.X.hc_attn_scale` | Attn HC scale |
| `layers.X.ffn_hc.base` | `layers.X.hc_ffn_base` | FFN HC base (reorder) |
| `layers.X.ffn_hc.fn` | `layers.X.hc_ffn_fn` | FFN HC function |
| `layers.X.ffn_hc.scale` | `layers.X.hc_ffn_scale` | FFN HC scale |

## Examples

### Expert Layer Keys
```
# Before
model.layers.0.mlp.experts.0.gate_proj.weight_packed
model.layers.0.mlp.experts.0.up_proj.weight_packed
model.layers.0.mlp.experts.0.down_proj.weight_packed

# After
layers.0.ffn.experts.0.w1.weight_packed
layers.0.ffn.experts.0.w3.weight_packed
layers.0.ffn.experts.0.w2.weight_packed
```

### HC (Hadamard Compression) Keys
HC keys are restructured to match BF16 format:

```
# Before
model.hc_head.hc_base
model.hc_head.hc_fn
model.hc_head.hc_scale
model.layers.0.attn_hc.base
model.layers.0.attn_hc.fn
model.layers.0.attn_hc.scale
model.layers.0.ffn_hc.base
model.layers.0.ffn_hc.fn
model.layers.0.ffn_hc.scale

# After
hc_head_base
hc_head_fn
hc_head_scale
layers.0.hc_attn_base
layers.0.hc_attn_fn
layers.0.hc_attn_scale
layers.0.hc_ffn_base
layers.0.hc_ffn_fn
layers.0.hc_ffn_scale
```

### Quantization-Specific Keys
Quantization metadata keys (e.g., `input_global_scale`, `weight_global_scale`, `weight_scale`) are preserved with their parent key transformation:

```
# Before
model.layers.0.mlp.experts.0.down_proj.input_global_scale
model.layers.0.mlp.experts.0.down_proj.weight_global_scale
model.layers.0.mlp.experts.0.down_proj.weight_scale

# After
layers.0.ffn.experts.0.w2.input_global_scale
layers.0.ffn.experts.0.w2.weight_global_scale
layers.0.ffn.experts.0.w2.weight_scale
```

## Usage

### Dry Run (Preview Changes)
```bash
python convert_safetensors_keys.py /path/to/model --dry-run
```

### Convert In-Place
```bash
python convert_safetensors_keys.py /path/to/model
```

**WARNING**: This modifies files in-place. While the script uses temporary files to avoid corruption, ensure you have enough disk space (approximately the size of the model directory).

## Technical Details

- Uses temporary files during conversion to avoid corruption
- Updates `model.safetensors.index.json` to reflect new key names
- Removes keys that don't exist in the BF16 model format
- Requires `safetensors` Python package
