# Sequential Pipeline #
The sequential pipeline is a data pipeline, primarily used for compressing models with the
[GPTQModifier](/src/llmcompressor/modifiers/gptq/base.py) or the
[SparseGPTModifier](/src/llmcompressor/modifiers/pruning/sparsegpt/base.py).

The pipeline exposes generic lifecycle events around each subgraph:

- `calibration_start` receives `subgraphs` and `dataset_args`.
- `sequential_epoch_start` receives the module list, executable `subgraph`,
  `subgraph_index`, current `activations` cache, and named `additional_activations`
  caches, before calibration forwards.
- `sequential_epoch_end` receives the module list, `subgraph`, and its index after
  all calibration batches, before propagating updated outputs.

Callbacks must not mutate the activation cache. References are valid for the
current stage; retain detached copies when data must survive cache updates.
Modifiers handle algorithm-specific work, including teacher target capture and
optimization, while the pipeline controls execution and propagation.

An optional `additional_dataloaders` mapping creates independent auxiliary
activation streams, e.g. `{"qad": qad_dataloader}`. These streams never trigger
calibration hooks and propagate only after all subgraph-end modifiers finish.
They require `propagate_error=True` and inputs compatible with the same traced
graph, but may have different batch counts, batch sizes and sequence lengths.
