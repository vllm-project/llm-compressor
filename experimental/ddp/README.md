## DP + AutoRound PoC

## Status
- [x] PoC implementation completed; functionality works.
  - AutoRound PR: https://github.com/intel/auto-round/pull/1407

```bash
CUDA_VISIBLE_DEVICES=0,1 python ddp_qwen3_example.py \
    --model Qwen/Qwen3-8B \
    --nsamples 128  \
    --iters 200 \
    --disable_torch_compile \
    --deterministic

```

## Next Steps
- [ ] Accuracy verification (align results with non-DP run)
- [ ] Quantization-time benchmark (single GPU vs DDP)
- [ ] UX alignment (APIs, config, logging)
