# Distributed FlexSmooth search: local validation

`transform/flex_smooth/distributed.py` provides the collective search used by
FlexSmoothModifier with `max_tokens=None`. The shared CPU/disk two-rank sequential
path now passes integration tests; see [the oneshot driver](oneshot_profile.md).
The optional token-ID/cap primitive below remains independently tested, not wired
to hooks. Local tests do not establish Ascend execution or real-model accuracy.

All ranks supply replicated weights from the same snapshot and disjoint local
activation rows. The primitive checks shapes/dtypes and input validity across
the process group; equality of weight contents is a caller precondition. It
communicates channel maxima, loss moments and integer metadata, never raw
activations or weights.

Each candidate uses the global activation channel maximum. Local squared error
and golden-output energy contribute to global mean moments; the final loss is
the ratio of their square roots. Averaging rank-local normalized losses would
weight ranks incorrectly and is not used. Empty ranks contribute zero moments
and still participate in every collective. All ranks use the same 21-alpha,
21-beta order, first-stage threshold and later-equal-wins policy as local search.

Proxy GEMMs, subtraction, square, square root and division use storage dtype.
Mean accumulation/reduction uses FP32 (FP64 for FP64 input), then casts back before
square root. Divide-before-sum avoids overflowing a sum whose mean is finite.
Storage-dtype square overflow remains nonfinite. Different GEMM/reduction order
can change loss values or a close candidate decision; universal bitwise equality
with single-process ModelSlim is not claimed. The existing local search path
retains its original arithmetic and source-differential tests.

An optional **global** token budget requires original global sample/token IDs.
The smallest N unique IDs are selected regardless of rank placement or local
row ordering; only shortlisted IDs are gathered. Rank-local batch numbers are
not valid global IDs. Duplicate candidate IDs, including sampler-padding
duplicates, are rejected. The caller must deduplicate and propagate stable IDs
through the actual dataloader/hook lifecycle before this can be integrated.
For a budget N each rank can retain only its smallest N IDs/rows; applying N
independently on every rank without global selection changes the sample set.

Local tests spawn real one- and two-process CPU Gloo groups, comparing 48 cases
per world size with the separately licensed, unmodified ModelSlim source:

- FP64, FP32, FP16 and BF16; three seeds; uneven/interleaved/empty-rank splits.
- Uncapped data and a five-token global cap, including reversed local row order.
- All 42 candidate losses, alpha/beta, applied scale and selected token counts.
- Rank agreement, equal-loss tie policy, all-zero and overflow candidates.
- Collective rejection of empty global data, NaNs, invalid/mismatched shapes or
  budgets, missing IDs, and local/cross-rank duplicate IDs.

All tested selections and scales match ModelSlim exactly. For two ranks, the
largest candidate-loss differences are FP64 `4.51e-17`, FP32 `5.03e-8`, and zero
in these FP16/BF16 cases. These are fixture observations, not dtype-wide guarantees.
The full local gate passes 218 tests. Run without any checkpoint:

```bash
python tools/run_quarot_checks.py --include-flex-smooth \
  --modelslim-source /path/to/local/msmodelslim \
  --report-dir /tmp/glm52-local-checks
```

Reports include `collective-search-{1,2}-rank.json`, reference hashes, per-case
loss differences, candidate grids and rank results. The communication test
rejects non-integer all-gathers so raw activation gathering cannot slip in.

The Modifier now agrees its plan at initialization and commits Flex mappings with
one read-completion and one write-completion status reduction. It does not use the
per-parameter [writeback prototype](distributed_offload.md). Global token caps are
deferred; first server runs use small datasets and no cap.
