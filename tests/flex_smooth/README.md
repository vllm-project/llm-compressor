# FlexSmooth correctness and lifecycle

`FlexSmoothModifier` is an independent modifier with GLM MLA norm-linear and OV
mapping. It searches alpha/beta using a symmetric INT8 reconstruction proxy and
fuses scales into floating weights. Quantization is a later recipe stage.

```python
from llmcompressor.modifiers.transform import QuaRotModifier, FlexSmoothModifier

recipe = [
    QuaRotModifier(block_size=32),
    FlexSmoothModifier(),
]
```

Pass this recipe, an initialized supported model and calibration data to `oneshot`.
Both basic and sequential pipelines are tested on a random official
`GlmMoeDsaForCausalLM`. No model weights or datasets are downloaded by this suite.
The only checkpoint saving/loading is of the generated tiny fixture.

Run the strict combined local gate from the repo root:

```sh
python tools/run_quarot_checks.py --include-flex-smooth --modelslim-source /path/to/msmodelslim --report-dir /path/to/report
```

The runner requires all L0–L3 tests to pass without skips and records source hashes,
dependency versions, JUnit, per-subgraph alpha/beta/scale, parameter errors, and
official tiny forward/reload errors. Use a supported installed LLMC CPU environment.

Coverage includes original GLM52 adapter mappings, all 42 candidate losses,
FP32/BF16 alpha/beta/scale/weight/output differential, strict FP64 float invariance,
untouched K rows, bias handling, grouped OV scale geometry, invalid inputs,
recipe round-trip, bounded caches, duplicate events, explicit all-invalid behavior,
CPU/disk offload, cleanup, QuaRot composition, and official GLM basic/sequential
calibration followed by native expert repack and local export/reload.

The external oracle executes unchanged selected definitions from separately licensed
ModelSlim. It adapts logging/context recording and single-rank expert range; it does
not emulate the full processor, NPU stack or distributed scheduler. The OV oracle
constructs a virtual V Linear around the selected rows for the original fusion code.
No ModelSlim algorithm implementation is copied into this repository.

Defaults retain every calibration token on CPU for reference parity. `max_tokens=N`
retains the first N tokens per subgraph and derives both search and scale statistics
from that subset; `diagnostics` reports seen/used counts. Fixed alpha/beta apply only
when both are specified; supplying either alone searches both, matching reference.

Every-candidate-nonfinite cases raise by default. `on_degenerate="identity"` explicitly
records an identity fallback, with no claimed optimum. Nonfinite input/weight data
still fails. This deliberate departure from the reference's infinite loss/default
parameters is separately tested. Model config records completion/failure and prevents
reapplication; unexpected mutation failures require a fresh model.

Initial limits: unsharded explicit GLM projections and expanded MLA V heads; no MTP,
mixed dtypes among norm consumers, loss masks, split-subgraph sequential partitions
or distributed execution. Grouped OV scale math is tested, but a grouped-KV model
adapter is not implemented. Dynamic mixed MXFP4/MXFP8 linear composition is also
tested on the official tiny GLM in basic/sequential pipelines, including exact
post-transform weight scales and explicit dynamic-input Q/DQ execution. The
standalone server header probe is tested with generated single/sharded checkpoints
and no site-packages. See `examples/glm52_precision/README.md` for the handoff.
The suite also checks actual mixed compressed export/reload in FP32/BF16 using a
scoped explicit-GLM expert constructor and the real HF quantizer. Packed weights,
E8M0 bytes, restored Q/DQ weights/scales, schemes, transform metadata and logits are
checked, including repeated-transform rejection. CT's BF16 decompression is followed
by explicit dtype normalization. Full FP32/BF16 mixed CPU/disk offload state matches
the resident run exactly; disk Q/DQ forwards use the sequential pipeline's cache
lifetime context. Existing config-resave regressions run in the combined gate.
See `examples/glm52_precision/runtime_limits.md` for the generic loader dependency,
decompression dtype and bare disk-forward limitations found by these tests.
Server kernels, KV-cache quantization and real-model L4/L5 remain separate gates.
No full-model accuracy claim follows from local tests.
