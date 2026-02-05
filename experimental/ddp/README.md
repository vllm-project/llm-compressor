## DP + AutoRound PoC

## Status
- [x] PoC implementation completed; functionality works.
  - LLMC PR: https://github.com/yiliu30/llm-compressor-fork/pull/18
  - AutoRound PR: https://github.com/yiliu30/auto-round-fork/pull/15

```bash
CUDA_VISIBLE_DEVICES=0,1 python ddp_qwen3_example.py --ddp --nsamples 128 --iters 100

```

## Next Steps
- [ ] Accuracy verification (align results with non-DP run)
- [ ] Quantization-time benchmark (single GPU vs DDP)
- [ ] UX alignment (APIs, config, logging)
