# Mixture of Experts (MoE) Compression with HOPE Expert Pruning

## Algorithm

HOPE (Higher-Order Pruning of Experts) was introduced in [Higher-order pruning
of experts in mixture-of-experts language
models](https://arxiv.org/abs/2609.18916). Like REAP (see
[`examples/reap_expert_pruning`](../reap_expert_pruning)), it reduces the
VRAM footprint of MoE models by structurally removing individual experts from
their MoE layers, using a calibration dataset to decide which experts to
remove. Where REAP scores each expert in isolation (the router-weighted mean
of its output norms), HOPE also accounts for the interactions between *pairs*
of experts.

During calibration, HOPE accumulates a per-layer `E x E` interaction matrix
`F`,

```
F_ij = mean(g_i * g_j * ||f_i||_2 * ||f_j||_2)
```

where the mean is over the calibration tokens which co-select experts `i`
and `j` (`F_ij = 0` for pairs which are never co-selected), `g` is the
router gate weight and `||f||_2` the expert output norm. The diagonal
`F_kk` is expert `k`'s self-importance: its mean squared gate-weighted
output norm, which recovers the squared REAP score up to a variance term.
The off-diagonal `F_ij` captures how much the pair contributes jointly when
co-selected. The prune-set `P` is chosen to minimize the interaction mass it
removes,

```
min_p  p^T F p    s.t.  p in {0, 1}^E,  sum_k p_k = |P|
```

where `p_k = 1` iff expert `k` is pruned. This provably minimizes an upper
bound on the squared output error introduced by pruning, and reduces to
REAP's independent expert ranking when the off-diagonal interactions are
zeroed. The binary program is solved through a continuous relaxation
(SciPy SLSQP) rounded to the `|P|` entries with the highest values, so the
`scipy` package must be installed (`pip install llmcompressor[hope]`).

Under group-limited routing
(`n_group > 1`, e.g. DeepSeek-V3), every group must stay routable, so HOPE
removes the same number of experts from every group, with one budget
constraint per group in the quadratic program.

## Usage

HOPE is useful for reducing the VRAM footprint of MoE models without
sacrificing model quality, and its advantage over first-order methods grows
with the pruning rate. In the paper's evaluation on three MoE models (up to
122B parameters), HOPE has the best average rank among the REAP, EAN, MAN
and Frequency baselines, and at 40-50% pruning (the abstract cites 50%) it
attains an average rank of 1.58 out of 5 methods (versus 2.42 for REAP),
with gains of up to 6.1% over REAP on agentic coding.
As with any calibration-based method, excess pruning degrades quality, and
the calibration data should cover the domains the pruned model will be used
for.

`HOPEPruningModifier` supports report-only mode with `sparsity=0.0`, which
does not need `scipy` (write per-expert statistics to a `.json` report
without touching the model), saliency visualization with
`tools/plot_reap_report.py`, group-limited routers such as DeepSeek-V3's, and
data-parallel calibration (see
[the distributed guide](../../docs/guides/big_models_and_distributed/distributed_oneshot.md#hopepruningmodifier)):

```python
recipe = HOPEPruningModifier(sparsity=0.25)
```

`sparsity` is the fraction of experts to remove from every MoE layer. The
router must still be able to select its top-K experts afterwards, so too
aggressive a value is rejected at initialization.

The report maps `count` (routed tokens per expert) and `saliency` (the
diagonal of `F`) to one inner list per MoE layer, the layout
`tools/plot_reap_report.py` reads (its plot title still says REAP). Unlike a
REAP report, it is not enough to reproduce the pruning decision, which also
depends on the off-diagonal entries of `F`: a tool which ranks experts by the
report's `saliency` column, such as the checkpoint-space `ExpertPruner`,
therefore falls back to a first-order ranking, not HOPE's prune-set.
`saliency` is also on a different scale from REAP's (the mean of the
squared gate-weighted norm, against the mean of the norm), so the two
reports' values are not comparable.

Calibration runs with `moe_calibrate_all_experts=False` (each expert runs
only on the tokens routed to it, which is all a pruning-only run needs) or
`True` (every expert runs on every token, as a quantizer combined with HOPE
may need for its statistics). Both execution paths produce the same
statistics.

### Token masking

When the calibration data provides a `loss_mask` field and is loaded with
`use_loss_mask=True`, only tokens where the mask is nonzero contribute to the
interaction statistics -- e.g. only assistant responses, excluding user
prompts and role headers (see `hope_moonlight_16b_with_masking.py`).
Masked-out tokens still flow through the model; they are simply excluded
from the accumulated statistics, exactly like the masking applied to the AWQ
activation statistics, the GPTQ Hessian and the REAP saliency statistics.

A masked-out token adds nothing to `sum_products`, `pair_count` or `count`,
so `F` is computed over the kept tokens only. The mask must cover exactly
the tokens of the batch a MoE layer saw; anything else raises an error
instead of misaligning the statistics. A layer whose tokens are all masked
out is skipped with a `Failed to calibrate` warning and left unpruned.

## Moonlight-16B example

`hope_moonlight_16b_with_masking.py` calibrates
[`moonshotai/Moonlight-16B-A3B-Instruct`](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct)
on the first 512 samples of `mlabonne/open-perfectblend` with token masking,
and prunes 25% of the experts in each MoE layer (64 -> 48). `loss_mask` is 1
only on assistant-response tokens; 24 of the 512 samples have no such token
and contribute nothing.
