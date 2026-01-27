## DP + AutoRound PoC

## Status
- [x] PoC implementation completed
  - LLMC PR: https://github.com/yiliu30/llm-compressor-fork/pull/18
  - AutoRound PR: https://github.com/yiliu30/auto-round-fork/pull/15

```bash
python ddp_qwen3_example.py --ddp --nsamples 128 --iters 100
```

## Next Steps
- [ ] Accuracy verification (align results with non-DP run)
- [ ] Quantization-time benchmark (single GPU vs DDP)
- [ ] UX alignment (APIs, config, logging)

## Notes
- Keep logs for both DP and non-DP runs to make accuracy comparison straightforward.
- Record hardware, batch size, and sequence length for each timing run so results are comparable.