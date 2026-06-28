# SafeTensors Key Name Conversions

This document describes the key name transformations applied by `convert_safetensors_keys.py`.

## Transformation Rules

| Original (Quantized) | Converted (BF16 Format) | Note |
|---------------------|------------------------|------|
| `head.weight` | `lm_head.weight` | LM head weight |
| `model.embed_tokens.weight` | `embed.weight` | Embedding layer |
| `model.layers.X.*` | `layers.X.*` | Strip `model.` prefix |
| `model.layers.X.input_layernorm.weight` | `layers.X.attn_norm.weight` | Attention normalization |
| `model.layers.X.mlp.experts.Y.gate_proj.*` | `layers.X.ffn.experts.Y.w1.*` | FFN gate projection |
| `model.layers.X.mlp.experts.Y.up_proj.*` | `layers.X.ffn.experts.Y.w3.*` | FFN up projection |
| `model.layers.X.mlp.experts.Y.down_proj.*` | `layers.X.ffn.experts.Y.w2.*` | FFN down projection |
| `model.hc_head.hc_base` | `hc_head_base` | HC head base (flatten structure) |
| `model.hc_head.hc_fn` | `hc_head_fn` | HC head function |
| `model.hc_head.hc_scale` | `hc_head_scale` | HC head scale |
| `model.layers.X.attn_hc.base` | `layers.X.hc_attn_base` | Attention HC base (reorder) |
| `model.layers.X.attn_hc.fn` | `layers.X.hc_attn_fn` | Attention HC function |
| `model.layers.X.attn_hc.scale` | `layers.X.hc_attn_scale` | Attention HC scale |
| `model.layers.X.ffn_hc.base` | `layers.X.hc_ffn_base` | FFN HC base (reorder) |
| `model.layers.X.ffn_hc.fn` | `layers.X.hc_ffn_fn` | FFN HC function |
| `model.layers.X.ffn_hc.scale` | `layers.X.hc_ffn_scale` | FFN HC scale |

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
