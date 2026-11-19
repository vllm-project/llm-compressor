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
projection of the experts-block input, transformed by the router's native
score function (`softmax`, `sigmoid`, or `sqrtsoftplus`, e.g.
DeepSeek-V4's `sqrt(softplus(z))`), which is read from the model config and
falls back to `softmax` when the architecture does not declare one. An
`e_score_correction_bias` (e.g. DeepSeek-V3, Hy3) steers which unselected
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

## GLM-5.3-Flash example

`glm53_flash_example.py` calibrates
[`RedHatAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)
in report-only mode (`sparsity=0.0`), writing per-expert statistics for the 42
backbone MoE layers without modifying or saving the model. The model is loaded
with `load_context(Glm5NextForConditionalGeneration, load_mtp=False)`: MTP
calibration is not supported, so the MTP layer (`layers.45`, one past the 45
backbone decoder layers) is left out of the report. Since the checkpoint's
experts are already NVFP4-quantized, `layerwise_decompression=True` decompresses
each layer just before calibrating it, and the run is single-process: the
decompressed expert weights (~600 GB) accumulate in host RAM, where a
distributed run would instead need that much `/dev/shm`.

`glm53_flash_prune_from_report.py` then prunes 20% of the experts directly
from the safetensors checkpoint with `ExpertPruner`, producing a
`GLM-5.3-Flash-NVFP4-Sparse20` model in `HF_HUB_CACHE`. The patterns passed to
`ExpertPruner.from_pretrained` exclude the uncalibrated MTP layer, and
`uniform=False` records the per-layer retained expert counts in
`quantization_config.moe_sparsity_config` while `n_routed_experts` keeps its
original value of 288, which still matches the untouched MTP layer.

## Nemotron-3.5-Super-VL example

`nemotron35_super_vl_example.py` calibrates
[`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16)
in report-only mode. Nemotron-3.5's MoE layers are *latent* MoEs (`nemotron_h`
with a `moe_latent_size`): each block projects its input down to 1024
dimensions for its experts while the router keeps scoring the full 4096-wide
block input. The modifier resolves the routing context from the activations
the router itself consumed (recorded by a pre-hook on the router), reads the
router's sigmoid score function from the architecture, and evaluates the
consensus, the residuals, and the deletion damage in the experts' latent space.

The saliency report is calibrated on the BF16 release because the
NVFP4-compressed-tensors release cannot be loaded for calibration yet: it is a
remote-code vision-language composition without a direct 2D expert load
mapping, and its quantization config lists exact module paths in checkpoint
naming (`language_model.backbone.*`) which do not match the module tree
transformers builds (`language_model.model.*`). The routing statistics are
shared between the releases (the router, the latent projections, and the
shared experts are unquantized in the NVFP4 checkpoint), so the report
transfers up to NVFP4 quantization noise on the expert outputs.

`nemotron35_super_vl_prune_from_report.py` then prunes 20% of the experts
directly from the
[`inference-optimization/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-compressed-tensors`](https://huggingface.co/inference-optimization/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-compressed-tensors)
safetensors with `ExpertPruner`, preserving the checkpoint's NVFP4 weights:
only expert tensors are dropped/renumbered and router rows (and their
`e_score_correction_bias`) sliced. The checkpoint's MTP layers live under a
separate `language_model.mtp.*` prefix, which the backbone patterns do not
match, so they are left untouched without extra pattern exclusions.
