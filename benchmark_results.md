## MSE Observer torch.compile Benchmark Results

| Metric | Baseline | Compiled | Delta |
|---|---|---|---|
| Time (s) | 28.3 | 28.1 | -0.9% |
| Peak memory (MB) | 2178 | 2133 | -2.1% |
| Quantized layers | 154 | 154 | - |
| Scales all_close | - | - | False |
| Max abs diff | - | - | 6.10e-05 |
| Max rel diff | - | - | 5.59e-02 |
| Matching layers | - | - | 26/154 |

**Speedup**: 1.01x

**Config**: model=TinyLlama/TinyLlama-1.1B-Chat-v1.0, dataset=open_platypus, samples=64, max_seq_length=384, atol=1e-05, rtol=0.001

**Mismatched layers** (128):
- `model.layers.0.self_attn.q_proj`
- `model.layers.0.self_attn.k_proj`
- `model.layers.0.mlp.gate_proj`
- `model.layers.0.mlp.up_proj`
- `model.layers.0.mlp.down_proj`
- `model.layers.1.self_attn.q_proj`
- `model.layers.1.self_attn.k_proj`
- `model.layers.1.mlp.gate_proj`
- `model.layers.1.mlp.up_proj`
- `model.layers.1.mlp.down_proj`
- ... and 118 more
