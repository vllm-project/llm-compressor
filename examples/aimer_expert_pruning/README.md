# Calibration-Free MoE Expert Pruning with AIMER

## Algorithm

AIMER (Absolute mean over root mean square IMportance for Expert Ranking) was
introduced in [AIMER: Calibration-Free Task-Agnostic MoE
Pruning](https://arxiv.org/abs/2603.18492). Like
[REAP](../reap_expert_pruning/README.md), it reduces the memory footprint of
Mixture-of-Experts models by structurally removing experts from their MoE layers.
Unlike REAP, it ranks experts without any calibration data, from a statistic of
their weights alone:

```
AIMER = sum(|w|) / sqrt(N * sum(w^2))
```

where `w` are all of an expert's weights (its gate, up and down projections) and
`N` is their total number. The score is the expert's mean absolute weight divided
by its root mean square weight, so it is independent of the weights' scale. As
in the paper, the experts with the highest scores in each layer are pruned.

## Usage

`AIMERExpertPruner` is a converter: it scores and prunes experts directly on the
safetensors checkpoint, shard by shard, without loading the model into memory
(see `olmoe_aimer.py`). Because no calibration data is needed, the ranking does
not depend on any dataset. `from_pretrained` reads every expert weight once to
score the experts, and `convert_checkpoint` reads the checkpoint again to write
the pruned copy.

Apart from its scoring, `AIMERExpertPruner` prunes exactly like `ExpertPruner`.
With `uniform=True` (the default, and what the paper evaluates) the same
fraction of experts is pruned from every layer. With `uniform=False` the
highest-scoring experts across all layers are pruned instead, and the per-layer
counts are recorded in the checkpoint's `quantization_config.moe_sparsity_config`.
This global allocation is inherited from `ExpertPruner` and has not been
validated against the paper. Group-limited routers only support `uniform=True`,
which drops the same number of experts from every group.

AIMER scores each expert's projection weights: tensors named `<proj>.weight`
(one per expert) or stacked `<name>_proj` parameters, as in the Hugging Face MoE
implementations. Projections named otherwise are not scored. Biases and other
expert tensors are pruned with their expert but not scored.

Every tensor matched by `expert_pattern` must have dtype F16, BF16, F32 or F64.
This rejects matched packed-integer and FP8 representations; it does not detect
all forms of quantization. The intended workflow is to prune before quantizing:
the pruned checkpoint can then be loaded and quantized with `oneshot` as usual.

As with any expert pruning, excess sparsity degrades model quality. Validate the
pruned model on the workloads that matter to you.
