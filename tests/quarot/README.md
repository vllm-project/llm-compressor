# QuaRot local correctness harness

Run from the repository root:

```sh
python -m pytest tests/quarot/test_rotation.py -q -p no:cacheprovider
```

Install the repository and its declared dependencies in an isolated CPU environment,
plus pytest. All tests now use normal public LLMC imports; runtime APIs are not stubbed.
No model downloads, NPU, GPU, or datasets are used. The only checkpoints saved or
loaded are tiny random fixtures generated within tests.

For L2 set `MODELSLIM_SOURCE` to a local, unchanged ModelSlim source checkout
(or use the workspace's sibling `reference/msmodelslim`). NumPy and packaging are
needed for the original reference seed routine.
The test adapter executes selected original function bodies directly from that
checkout and records source SHA256 hashes. It does not vendor reference algorithms.
Missing reference source is an explicit skip for ordinary pytest runs. The gate
below fails for missing source, missing dependencies, or any skipped tests:

```sh
python tools/run_quarot_checks.py --modelslim-source /path/to/msmodelslim --report-dir /path/to/reports/run-001
```

The gate writes JUnit, environment/commit metadata and per-parameter error reports
for initial, norm fusion, embedding and all four rotation stages. Cases cover
block_size=32 and reference default (-1), q_lora_rank=32/64, FP32/BF16 storage,
and three matrix seeds. FP64 forward algebra uses a separately constructed FP64 Q.

Oracle boundary: selected original functions and GLM52 adapter methods are executed,
not the entire ModelSlim processor/import graph. NPU discovery and app initialization
are not run. CPU seed behavior is selected; logging uses Python logging; exception
numeric codes are irrelevant to numerical comparisons. Non-power-of-two asset lookup
is unsupported. Only named MTP-only targets are removed from the decoder reference
mapping. Every other target must resolve. No reference implementation is vendored.

L3 covers public Modifier events, recipe YAML round-trip, duplicate-application guards,
tied CPU embeddings, real CPU/disk offload caches and runtime failure state. Official
`GlmMoeDsaForCausalLM` random tiny configs exercise full/shared and full/full indexers
through `oneshot`, native expert repacking, local save and reload. This complements
the controlled mathematical fixture `tiny_glm.py`; neither predicts real accuracy.
On Windows run in a context that can access pytest's private temporary directories.

Current limits: power-of-two blocks, explicit unsharded experts (including LLMC's
linearized experts), no MTP or online rotations. Shared disk-backed embedding
parameters must be untied before offloading. Invalid topology fails before mutation;
unexpected I/O/OOM failures mark config state `failed` and require a fresh model,
without allocating a model-sized rollback copy. Persisted `quarot_config` metadata
prevents accidentally applying a new modifier to an already transformed model.

Non-power-of-two full bases, distributed execution, production MLA decode caches,
FlexSmooth and mixed MXFP composition remain separate gates. No real-model accuracy
or deployment claim follows from these tests.
