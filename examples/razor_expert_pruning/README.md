# Mixture of Experts (MoE) Compression with RAZOR Expert Pruning

## Algorithm

RAZOR was introduced in [RAZOR: Pruning Replaceable Experts in
LLMs](https://arxiv.org/abs/2609.30465). Like REAP (see
[`examples/reap_expert_pruning`](../reap_expert_pruning)), it reduces the
VRAM footprint of MoE models by structurally removing individual experts from
their MoE layers, using a calibration dataset to decide which experts to
remove. Where REAP scores each expert in isolation (the router-weighted mean
of its output norms), RAZOR scores each expert by the damage its *deletion*
would do to the layer's output, which also depends on how the surviving and
replacement experts compensate for the removed contribution.

Writing the routed mixture (the *consensus*) of a token as

```
c(x) = sum_{i in S(x)} w_i(x) * f_i(x)
```

over the router-selected experts `S(x)`, their (renormalized) mixture weights
`w_i(x)` and outputs `f_i(x)`, the consensus residual of expert `i` is
`r_i(x) = f_i(x) - c(x)`. Deleting expert `i` renormalizes the surviving
weights and, when the router refills the vacated slot, promotes the
highest-ranked unselected expert `r` with pseudo-weight
`w_r(x) = s_r(x) / sum_{j in S(x)} s_j(x)` (its score over the selected-score
sum). The exact token-level output change of a single deletion under this
counterfactual is

```
delta_i(x) = ||w_i(x) * r_i(x) - w_r(x) * r_r(x)||_2 / (1 - w_i(x) + w_r(x))
```

RAZOR aggregates this damage by conditional root mean square,

```
S_j = sqrt(sum_t delta_j(t)^2 / max(n_j, 1))
```

over the `n_j` calibration tokens routed to expert `j`, and removes the
lowest-scoring experts in each layer, up to the requested sparsity. All of it
uses forward computation alone: no gradients, subset search, or recovery
training. The `criterion` argument of `RazorPruningModifier` selects the
token-level damage quantity: `"refill"` (the default, RAZOR proper, which
credits the replacement expert), `"rcs_loo"` (the exact single-deletion change
when the vacated slot is *not* refilled), or `"rcs"` (the plain
routing-weighted consensus residual).

The router scores are recomputed each batch from the router's linear
projection of the experts-block input, transformed by the router's score
function (`softmax`, `sigmoid`, or `sqrt_softplus`). When left unset, the
score function is inferred from the model config, falling back to `softmax`.
An `e_score_correction_bias` (e.g. DeepSeek-V3, Hy3) steers which unselected
expert is promoted -- matching the router's selection rule -- but is not
added to the mixture weights. Under group-limited routing (`n_group > 1`,
e.g. DeepSeek-V3), the promoted expert is approximated by the globally
highest-ranked unselected expert, and the per-group expert budget stays
balanced as in REAP.

## Usage

RAZOR is useful for reducing the VRAM footprint of MoE models without
sacrificing model quality. In the paper's evaluation on GLM-4.7-Flash and
Qwen3.6-35B-A3B at 25% and 50% expert removal, it attains the highest
nine-task macro average among frequency, activation-norm, and REAP baselines
in all four model-budget settings, gaining 2.12-5.59 points over REAP, and
also lowers reverse KL in all four. As with any calibration-based method,
excess pruning degrades quality, and the calibration data should cover the
domains the pruned model will be used for.

`RazorPruningModifier` supports everything REAP's modifier does: report-only
mode with `sparsity=0.0` (write per-expert statistics to a `.json` report
without touching the model), pruning from that report directly in
checkpoint space with `ExpertPruner` (see
[`examples/reap_expert_pruning/qwen3_prune_from_report.py`](../reap_expert_pruning/qwen3_prune_from_report.py),
whose report format is shared), saliency visualization with
`tools/plot_reap_report.py`, and group-limited routers such as DeepSeek-V3's:

```python
recipe = RazorPruningModifier(sparsity=0.25, criterion="refill")
```

Calibration runs with `moe_calibrate_all_experts=False` (the experts block
evaluates only the routed experts, and the modifier itself evaluates the one
additional expert per token that the refill criterion needs) or `True` (every
expert runs on every token, and the refill expert's output is already
available). Both execution paths produce the same saliency.
