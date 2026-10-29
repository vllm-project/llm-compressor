# W4A16 iMatrix ablation

`manifest.json` defines the requested 48-point sweep: four models, six error
powers, and the two iMatrix objectives. Both objectives use W4A16, group size
64, `imatrix_mse`, patience 0, grid 100, and the same 512-example
PerfectBlend calibration slice at sequence length 2048.
Each variant uses calibration seed 42 so the same sample order feeds every
observer configuration.

The group-absmax objective weights each element in a quantization group by
`sqrt(w**2 + group_absmax**2)` in addition to activation importance and
powered quantization error. The baseline objective omits that factor.

Run the matrix with:

```bash
python examples/imatrix/ablation/orchestrate.py
```

Quantization jobs use `/home/HDCharles/rhdev`; serving and the lm-eval and
lighteval harnesses use `/home/HDCharles/vllm`. The runner reserves up to
eight GPUs through `chg`, runs 100-example smoke checks for each model profile,
then evaluates GSM8K Platinum, GPQA Diamond, and Math-500 with the registry's
three repetitions. Evaluation is staged globally by seed: every eligible
checkpoint completes the first seed before the second-seed pass begins, then
the third. The aggregate CSV/JSON matrix is refreshed after each pass so early
results are available before all repetitions finish. MMLU is excluded. It checks GPU
use and worker logs every five minutes. Checkpoints, per-task logs, manifests,
and the aggregated CSV/JSON results are written under
`/mnt/data/engine/HDCharles/experiments/w4a16-imatrix-ablation-2026-10-02/`.
