# qwen3b Benchmark Results

## Environment
- PyTorch: 2.9.1+cu128
- GPU: Tesla V100-SXM2-16GB
- Git: dd221a45

## Results

| Mode | First Run | Warm Avg | Stddev | Compile OH | Speedup | Peak Mem | Recompiles | max_abs_diff |
|------|-----------|----------|--------|------------|---------|----------|------------|--------------|
| baseline | 14m 33.4s | 14m 39.3s | 8.4s | - | 1.00x | 2.36 GB | - | - |
| compiled | 8m 32.5s | 7m 57.6s | 0.5s | 34.9s | 1.84x | 2.36 GB | 0 | N/A |

## Variance Statistics

| Mode | Mean | Stddev | CV | Min | Max |
|------|------|--------|----|----|-----|
| baseline | 14m 39.3s | 8.4s | 1.0% | 14m 33.4s | 14m 45.3s |
| compiled | 7m 57.6s | 0.5s | 0.1% | 7m 57.2s | 7m 58.0s |

## Compile Statistics

| Metric | Value |
|--------|-------|
| Graphs compiled | 2 |
| Graph breaks | 0 |
| Frames compiled | 2 |

