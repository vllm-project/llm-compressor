# Design note: meta initialization versus lazy scope initialization

**Status:** Draft for discussion
**Snapshot:** 2026-11-06

## The decision

There are two different questions:

1. **When do modifiers attach schemes, observers, hooks, and transform state?**
2. **When do real weight tensors get loaded, decompressed, transformed, and compressed?**

[llm-compressor#3342](https://github.com/vllm-project/llm-compressor/pull/3342) answers the first question with per-scope initialization: set up a subgraph when it is onloaded. [llm-compressor#3462](https://github.com/vllm-project/llm-compressor/pull/3462) uses meta tensors to create the future linearized MoE structure and initializes quantization state before its weights are materialized. The second approach can catch structural problems earlier, but it is not a free or general replacement for the first.

**Recommendation:** keep #3342's per-scope setup for real, stateful modifier work. Add a whole-model, read-only preflight using module/checkpoint metadata and meta shapes. Use that preflight to catch deterministic configuration, mapping, and shape errors before the first calibration forward. Do not run every modifier's full initialization on meta by default.

## What the two approaches mean

### Full meta initialization

The model graph and its parameters are created on meta, modifier initialization runs against that graph, and actual checkpoint tensors are later copied into the same modules. In #3462, the linearized MoE shells are created on meta, a model-specific expert-copy path is exercised for shape validation, and the source experts are retained until their owning scope is onloaded. The toy test also initializes a QuantizationModifier on meta and checks that hooks survive materialization.

Meta tensors have shape and dtype but no stored values. This can support structural validation, but every initializer must be safe when values are absent. Materialization must also preserve any module/parameter objects or observer state that modifiers retained.

### #3342 per-scope lazy initialization

Recipe/config validation happens up front. Module schemes, observers, hooks, and calibration state are then created for the current scope after onload (and, when enabled, decompression). Scope end applies qparams/transforms, cleans up hooks, and lets the pipeline propagate, compress, and offload.

This matches layerwise execution. The tradeoff is that checks performed only during scope setup can fail late. In the current branch, for example, group-size divisibility is checked while quantization is initialized for the current scope.

### Recommended hybrid

Create enough whole-model structure to describe the final logical modules, then run pure validators against that structure and the checkpoint metadata. Keep weight-dependent work and modifier state changes lazy.

For lazy MoE linearization, this can mean creating the meta linearized shells and validating every expert's source-to-destination shape mapping up front, while copying real weights only when the scope is onloaded. It does **not** require attaching every observer and hook to every module before the sequential loop.

## Is full meta initialization simpler and free of issues?

No. It is simpler only for code that depends on module names, tensor shapes, dtypes, and configuration. It becomes more complex as soon as initialization reads tensor values or later materialization must preserve initialized state.

The #3462 approach already needs special handling: retain the fused source module outside the registered tree, include it in the subgraph's staging/onload set, materialize the destination, preserve non-meta qparams and parameter identities, restore zero points, then release the source. The current diff resets zero points by scanning parameters; configurations can register them as buffers, so those need a defined reset too. This is a useful prototype for MoE conversion, not evidence that arbitrary modifier initialization works on meta. The PR explicitly leaves initializers that inspect real weight values untested.

One important stacking detail: #3462 demonstrates meta quantization initialization against the lifecycle that existed on its branch. With #3342, `QuantizationModifier.on_initialize` no longer attaches schemes to the whole model; actual scheme application and divisibility validation happen in `_prepare_sequential_epoch` for each scope. Therefore, adding #3462's meta MoE shells to #3342 does **not** by itself make divisibility or other per-scope checks run globally. That requires a separate whole-model validator or a deliberate all-modifier initialization pass on the meta graph.

Meta also saves tensor storage, not the Python module graph or all per-module observer/hook objects. A large MoE can still have substantial construction and metadata overhead. A meta operation may also be unsupported if code reads values (for example, a scalar check) or calls an operation without a meta implementation.

### Concrete incompatibilities in the #3462 approach

These are separate from the general fact that meta tensors have no values:

| Area | Concrete constraint |
| --- | --- |
| **Compressed source experts** | `meta_from_experts_module` creates dense 2D `Linear` destinations and its materializer copies only fused `weight` and optional `bias`. It does not split packed tensors or transfer the source quantization scales/zero-points. In the #3462 pipeline hunk, the pending source is materialized into the destination before the normal per-scope decompression call, and that normal call receives the registered destination modules, not the pending source. Therefore packed fused experts must be decompressed while the source is onloaded and **before** `copy_from_experts_module`, or this input format must be rejected. This is the main layerwise-decompression incompatibility. |
| **Meta qparams and buffers** | `to_empty` gives storage to all meta parameters and buffers, but does not restore their initialized values. The prototype saves/restores tensors that were already non-meta and resets parameter names ending in `_zero_point`; it does not reset zero-point buffers or define initialization for arbitrary modifier buffers. The #3462 review specifically called out zero-point buffers. If meta initialization attaches observers, caches, or other initialized buffers, each needs an explicit preserve, reset, or defer rule. |
| **Value-mutating setup** | Norm conversion (`weight = 1 + weight`), embedding centering, norm fusion, weight transforms, decompression, calibration, and compression need real values. Running them on meta either fails, computes only placeholder metadata, or has its result overwritten when checkpoint values are copied in. These steps must run after source materialization, in a deliberate order. |
| **Offload dispatch** | A meta destination has no offload cache containing checkpoint values. #3462 adds a special `set_onload_device_for_pending_moes` path that temporarily hides those destinations from generic dispatch, and manually adds the unregistered fused source to staging/onload. Generic device dispatch and subgraph staging therefore cannot simply treat the meta destination as an ordinary offloaded module. |
| **Model traversal and saving** | The pending fused source is deliberately kept outside `named_modules()` and `state_dict()`. Generic modifier scans, checkpoint coverage checks, and save code cannot see it. The PR explicitly materializes MoEs left out of traced subgraphs before finalization; exception cleanup must also release pending sources, and finalization must guarantee that no meta parameters remain. |
| **What the shape probe proves** | Calling the model-specific `copy_from_experts_module` on meta catches incompatible tensor shapes and exercises the slice/transpose code. It cannot verify that same-shaped checkpoint keys have the intended semantic layout, that values are finite, or that the checkpoint contains the expected data. Those still need load-time/runtime checks. |
| **PyTorch compatibility** | The PR's parameter-identity path uses `torch.__future__.get_swap_module_params_on_conversion` and `torch.utils.swap_tensors`; a review noted these APIs need a version guard if support below PyTorch 2.3 is required. This checkout declares `torch>=2.10`, so that is only a concern if the supported minimum is lowered. |

So “full meta initialization” can mean “construct the final module tree and allocate shape-only modifier state before onload.” It cannot mean “finish all modifier preparation and all pipeline operations on meta.” With #3342, the useful combination is meta structure plus read-only structural preflight, followed by per-scope stateful setup on real tensors.

## What can happen at each pipeline phase?

| Operation | Whole-model meta initialization | #3342 per-scope initialization | Recommended placement |
| --- | --- | --- | --- |
| **Build module graph / resolve names** | Works if model construction is meta-safe. A meta linearized MoE shell exposes the future expert projection names. | Sees only modules that exist when each scope opens. Lazy linearization can hide future projection names from global target matching. | Build a structural plan before calibration. Use meta shells or source/checkpoint metadata for modules created by lazy linearization. |
| **MoE linearization and mapping validation** | Can run the model-specific slicing/transposition path against meta shapes. #3462 does this for each expert. It cannot verify copied values. | Actual linearization can happen with the scope onloaded, but mapping/layout errors are discovered at that scope's turn. | Validate every source-to-logical-projection shape mapping up front on meta; copy real expert weights when their scope loads. Specify how precompressed fused experts are decompressed before slicing. |
| **Quantization target and group-size validation** | Can validate all scopes if quant schemes and final logical module shapes are available on meta. It need not run observers to check input-feature divisibility by group size. | The current scoped check sees one scope at a time. A bad final layer may fail after earlier layers have been calibrated or written. | Resolve schemes from the recipe and validate every logical target shape before the loop. Keep the scoped check as a runtime assertion. |
| **Onload / offload** | A meta placeholder contains no checkpoint data and is not itself an offload cache. The loader needs a mapping from each placeholder to its source tensor. #3462 adds pending-source bookkeeping and custom onload helpers for this. | Uses the ordinary scope onload/offload path and naturally bounds live weights. New modules created during lazy linearization still need their offload settings propagated. | Keep a clear source-to-destination mapping. Onload only the owning scope and any backing source tensors; preserve device/offload metadata when replacing or adding modules. |
| **Layerwise decompression** | Can validate format/config and logical shapes from metadata. It cannot reconstruct packed weights without the stored values. | Decompresses the real current scope after onload, before modifiers need dense/logical weights. This is the natural execution point. | Preflight metadata and shape compatibility globally; decompress actual values per scope. If the source MoE is packed, define whether to decompress it before linearization. |
| **Offset-norm calibration patch** | Can identify registered norm types and validate converter coverage. Applying one plus the offset weight on meta gives no usable calibrated weight; materialization would replace its absent value. | The current scoped conversion runs after onload/decompression, patches the forward path, then restores the offset convention before compression/offload. | Keep the numeric conversion per scope. Preflight only converter registration, module scope, and shared/tied-parameter constraints. Always restore in cleanup paths. |
| **Calibration, observers, optimization** | Cannot collect meaningful activation/Hessian statistics or evaluate data-dependent checks without real tensors and calibration samples. Some hook objects can attach to meta modules, but they do not make those checks early. | Runs against real weights and inputs in the current scope. Modifier-specific state can be released at scope end. | Keep this lazy. Failures based on actual values or observed statistics remain runtime failures. |
| **Layerwise compression/repacking/save** | Can validate support, configuration, and key conversion rules. It cannot pack or write meaningful weights from meta placeholders. | Compresses/re-packs real tensors after scope work and propagation, then writes/offloads them. | Validate format and save-path compatibility globally; do the actual packing and writes per scope. Use an incomplete marker or transactional output so late failures cannot look like a complete checkpoint. |

The norm ordering is important: the current scoped offset-norm code converts the loaded weight to standard-norm semantics before modifiers run, keeps the same module/parameter references where possible, and restores the offset representation after modifier work and propagation. It must not be moved wholesale to meta initialization.

## What can fail late under #3342?

The distinction is **late versus skipped**. Most per-scope setup errors still happen before that scope's calibration forwards, but only after earlier scopes may already have calibrated, transformed, compressed, or been streamed to disk. A smaller set of checks now runs only at finalization, and some target/mapping misses are warnings or silent no-ops rather than exceptions.

### Validation inventory

| Check or failure | Behavior with per-scope initialization | Can a preflight catch it? |
| --- | --- | --- |
| **Recipe/schema checks**: invalid field types, quantization-argument combinations, bad `start`/`end`, and modifier-specific option constraints | These generally remain in recipe/Pydantic validation or `on_initialize`; they are still early. GPTQ actorder conflicts, AutoRound/QAD pipeline restrictions, SmoothQuant data requirements, and expert-sparsity arithmetic checks are examples. | Already checked early. Do not duplicate these in the meta path. |
| **Target type and quantization setup** | `apply_quantization_config` now runs for a scope. Unsupported target module classes, unsupported attention/KV-cache combinations or missing attention cache setup, and shape-dependent qparam initialization can fail when the affected scope is prepared. Observer registry lookup/observer construction also occurs at scope start, so an invalid observer name or unsupported observer/args pair can fail there. Most schema errors (including static TOKEN quantization) are already rejected while validating `QuantizationArgs`; the lower-level initializer has a duplicate guard. `match_named_modules(..., warn_on_fail=True)` warns about globally unmatched patterns; a warning is not proof that every intended logical module was initialized. | Yes, for all final logical targets, their module types, resolved scheme, and observer registrations. Keep the runtime initializer as a guard. |
| **Weight group divisibility** | The explicit `GROUP`/`TENSOR_GROUP` weight-column check is now scoped. It can reject a late layer after earlier scopes ran. The current check does not flag BLOCK quantization: its scale shape uses ceil/padding semantics. | Yes, from each final logical weight shape and resolved scheme. |
| **Fused `TENSOR_GROUP` scope closure** | Fused sets such as Q/K/V or gate/up are checked when a scope is initialized / its observers are fused. If the selected sequential targets split a fused set, the first intersecting scope raises. This is not a missing mathematical qparam check; it is a scope-plan constraint. | Yes, after partitioning, before the first calibration forward. Validate that each fused set is wholly inside one scope. |
| **Lazy MoE target and source mapping** | If logical expert projections do not exist until onload, target matching, shape mismatches, and unsupported model-specific copy paths are discovered at that MoE's turn. A meta shell moves shape and module-name checks earlier, but not value/layout correctness. | Yes for module existence, names, shapes, and supported converter registration. Compare source checkpoint keys/shapes to destination mappings. |
| **AWQ mappings** | Model-wide mapping resolution is also used for memory estimates, so several structural errors are already early. Scope closure, duo-scaling compatibility with the quant scheme, and the MoE/token-mask incompatibility are checked during scope setup. Missing/incompatible balance layers can be warned-and-skipped. Unconsumed cached activations are checked at finalization. | Yes for mapping closure, quant-scheme compatibility, activation hook target existence, and loss-mask/MoE incompatibility. Whether an activation was actually observed/consumed remains a runtime check. |
| **SmoothQuant mappings** | Mapping names and shapes are resolved early for memory estimates, but a mapping split across scopes raises only when an intersecting scope is initialized. Incompatible/empty mappings may warn and be skipped; leftover scales are rejected at finalization. | Yes for mapping coverage, shape compatibility, and scope closure. Runtime must still confirm every expected activation produced usable statistics. |
| **SpinQuant / QuIP transforms and norm pairs** | Factories/configuration are created early, but target application and shape-sensitive transform construction occur per scope. SpinQuant's norm-plus-linear scope check also runs at scope preparation. A bad late transform dimension can therefore fail after earlier scopes changed weights. There is also a concrete custom-config gap in the reviewed #3342 code: `SpinQuantModifier.on_initialize` returns early when `transform_config` is supplied, before populating `_transform_factories`; the scoped call then creates a fresh factory per scope. Random or paired transforms split across scopes can therefore diverge without an exception. Some unmatched transform patterns may also warn or do nothing. | Yes for target resolution, transform dimensions, paired/inverse mapping closure, and scope assignment. Always initialize and reuse factories, including for user-supplied configs. Keep numerical application on real weights. |
| **GPTQ has no usable target** | The previous “unable to find any modules to quantize” check at calibration start is now accumulated across scopes and raised from `on_finalize`. A wrong target/config can therefore consume the full run before failing. | Yes: require at least one final logical weight target that can be calibrated by GPTQ. |
| **AutoRound target/config support** | Per-layer scheme conversion and optimizer construction happen when a scope ends. Unsupported activation/weight schemes, output activation quantization, or missing captured inputs can fail only after that layer's calibration; a scope with no decoding layer returns without an optimization error. | Yes for target presence, scheme support, and optimizer option compatibility. Missing actual captured inputs remains a runtime/tracing check. |
| **QAD has no preceding weight quantizer / did not optimize every target** | The no-quantized-module check is no longer effective at `on_initialize` for the sequential path (the callback receives a module list); the final set comparison can fail only at finalization. | Yes for recipe order and scope containment. Whether each target was actually visited and optimized still needs a runtime completion check. |
| **Other incomplete calibration** | Expert sparsity can warn and mark uncalibrated MoE layers as unpruned instead of failing. Quantization/transform target misses can also be warnings/no-ops. `end_calibration` handles observers that saw data, but there is no single pre-run assertion that every intended target got the expected statistics. | Preflight can prove target coverage and expected scope ownership. Runtime should report unvisited targets and missing statistics explicitly. |
| **Offset-norm converter coverage** | `prepare_norms_for_calibration` silently skips class names absent from its registry. A missing unit-offset norm implementation can therefore let AWQ/SmoothQuant/SpinQuant alter weights under the wrong norm convention without an error. This is a correctness gap, not caused by #3342; issue #3429 documented this failure mode for several Transformers norm classes. Keep registry aliases covered by tests and add a preflight check. | Yes, if the model's norm types and offset semantics are known. Numeric `1 + weight` conversion still stays after onload. |
| **Weight/data-dependent checks** | Finite values, calibration-statistic quality, Hessian/optimization failures, and actual kernel/data-path errors require loaded tensors and forwards. They remain runtime failures even with global meta setup. | No shape-only preflight can prove these. A streamed validation-only pass can catch some checkpoint-value issues, at the cost of another read. |
| **Automatic grouping estimates** | This is not a validation check, but it can become wrong if targets are created after `on_initialize`: quantization memory estimates cache the modules visible at that time. Lazy MoE projections must exist in the plan before automatic grouping, or estimated memory may be low and cause an OOM later. | Yes, if grouping consumes the final logical module plan and its modifier scratch estimates. |

### Quantization divisibility

Suppose most linear layers have 4096 input features and a later layer has 4100, while the selected group size is 128. The first layers pass; the later layer fails because 4100 modulo 128 is nonzero. A per-scope validator catches this when that later scope is prepared.

This check needs the resolved quantization scheme and the **logical** weight shape. It does not need weight values. A whole-model preflight can compute it before calibration, provided it knows the final shape after MoE linearization and after interpreting any compressed-tensor metadata. The runtime check should remain as a guard against divergence.

### Mapping errors with lazy linearization

Some mappings target the linear projection modules inside each expert. In a fused checkpoint those modules may not exist until linearization. If linearization waits until onload, global matching against the original tree cannot prove that every future projection exists or has a compatible shape. The first bad MoE scope may fail after earlier dense/MoE scopes have run.

The meta-shell route addresses the structural part: create the future module tree and exercise the copy/slice mapping on meta tensors for all experts. A metadata-only resolver is another option if the checkpoint index and architecture config fully describe the same mapping. Neither route proves the actual values are correct.

The same distinction applies to AWQ/SmoothQuant balance mappings, SpinQuant/QuIP paired transforms, and norm mappings: resolving names and dimensions is preflight work; collecting calibration values and applying weight transforms remains scope work.

## Specific implementation proposal

1. **Keep #3342's runtime lifecycle.** Continue attaching observers/hooks and running calibration work only after the scope is onloaded and any required decompression/materialization is complete.
2. **Add a read-only preflight hook.** Give modifiers a validator that receives the model/config and a CalibrationPlan. It may resolve names, inspect shapes/metadata, and return errors. It must not attach hooks, mutate weights/qparams, run calibration, or write output.
3. **Build the plan before the first calibration forward.** Include all traced scopes, persistent modules, final logical module names, and the order of linearization, decompression, norm conversion, compression, and saving. Current auto-partitioning is done after persistent modifier setup because memory estimates use modifier state. The clean design moves planning earlier by making estimates config-driven. An interim preflight can run just after partitioning and before any subgraph calibration; because persistent hooks may already be attached, it must unwind those hooks and norm changes if validation fails.
4. **Start with deterministic failures.** Validate global target coverage, group-size divisibility for every logical quantized layer, fused/tensor-group closure, AWQ/SmoothQuant mapping closure, SpinQuant/QuIP transform pairing, norm converter registration, and lazy MoE source-to-destination shapes.
5. **Close the known implementation gaps.** Initialize/reuse SpinQuant factories even for a caller-supplied `transform_config`. For packed MoE inputs, decompress the pending fused source after onload and before slicing, or reject that combination clearly. Define materialization for every registered parameter and buffer, and assert that no meta parameters or pending sources remain before finalization/save. Add completion checks for GPTQ, QAD, AutoRound, and expected calibration statistics.
6. **Use meta for structure, not fake data.** The MoE meta shell can make future projections visible and validate their shapes. Do not run numeric norm conversion, real decompression, calibration, or compression on those placeholders.
7. **Keep the checks at runtime too.** After a scope is materialized, repeat inexpensive structural checks before calibration. Report both the module path and the owning scope.
8. **Separate truly data-dependent checks.** Finite-value checks, calibration statistics, Hessian/activation behavior, and numerical/kernel failures cannot generally be proven from meta shapes. Keep them in the scope that has the real values.

The intended execution order is:

1. Trace and determine scopes.
2. Build the structural/logical module plan, including meta linearized MoE shells when needed.
3. Run the whole-model read-only preflight.
4. For each scope: stage/onload; decompress the source if required; materialize lazy linearized weights; prepare offset norms; initialize modifier state/hooks; calibrate and apply transforms; run propagation/error measurement; restore norms; compress/repack/write/offload.

The order between decompression and MoE slicing depends on the source representation. For dense fused experts, create the logical projections and copy the current scope's values. For packed fused experts, decompress the source representation before slicing unless the decompressor explicitly supports the packed projection layout.

## Modifier-specific constraints

- **QuantizationModifier:** target coverage and group divisibility are deterministic preflight checks. Observer creation, activation collection, qparam updates, and freezing stay per-scope.
- **GPTQ and pruning/sparsity modifiers:** validate target/scheme compatibility, group size, act-order settings, expert layout, and distributed scope order early. Hessian/saliency values and quantize/prune work require loaded tensors and real calibration inputs.
- **AWQ and SmoothQuant:** resolve all balance/smoothing mappings and determine their scope closure early. If a pair spans scopes, either reject it before work or explicitly retain/cache the needed activations and weights across scopes. Do not discover an unsupported pair midway through the run.
- **SpinQuant and QuIP:** validate pair dimensions and inverse relations globally. Reuse the same transform factory/seed state across scopes; independently creating factories can make split pairs inconsistent.
- **AutoRound:** validate target support, optimizer options, memory estimates, and calibration-data reuse up front. Its optimization uses real weights and forwards, so meta initialization cannot perform it. Preserve its layer-local state and independent-layer error-propagation behavior.
- **QAD:** validate ordering and target relationships early. Its end callback may need the quantizer's observers and cached teacher/reference values, so cleanup must remain after all modifiers process scope end.
- **Offset norms:** validate converter coverage and module ownership in preflight. Convert actual weights only after onload/decompression; restore on every success or failure path before compression/offload.

## Other alternatives

- **Checkpoint-header-only preflight:** inspect model config, safetensors names/shapes, and quantization metadata without constructing all target modules. This can be cheaper than a full meta model, but may not reproduce Transformers' model-specific MoE mapping or modifier target resolution.
- **Streamed validation-only pass:** onload/decompress each scope once without calibration, inspect actual values, then run the real pass. This catches more value-dependent problems at the cost of reading the model twice.
- **Transactional/resumable output:** does not move validation earlier, but limits damage from failures that are inherently data-dependent. It is especially useful with streaming save.

## Conclusion

Full meta modifier initialization is **not** a simpler, issue-free replacement for #3342. It can validate shape/configuration earlier, and #3462 shows a credible way to use it for MoE structure and basic quantization setup. It cannot perform the real onload, decompression, norm-weight conversion, calibration, or compression steps, and it introduces materialization/state-preservation requirements.

The specific change to pursue is: **retain per-scope lazy modifier setup; add a global read-only preflight over the final logical scope plan; use meta MoE shells where they are needed to expose future module structure; keep runtime assertions and real value operations in the scope path.**

## References

- [llm-compressor#3342: Refactor initialization + calibration start](https://github.com/vllm-project/llm-compressor/pull/3342)
- [compressed-tensors#930: Utils updates for sequential pipeline](https://github.com/vllm-project/compressed-tensors/pull/930)
- [llm-compressor#3234: Lazy linearization and repacking](https://github.com/vllm-project/llm-compressor/pull/3234)
- [llm-compressor#3462: Prototype meta MoE linearization for sequential calibration](https://github.com/vllm-project/llm-compressor/pull/3462)
- [llm-compressor#3162: Layerwise compression and decompression](https://github.com/vllm-project/llm-compressor/pull/3162)
- [compressed-tensors#867: Layerwise decompression and compression](https://github.com/vllm-project/compressed-tensors/pull/867)
- [llm-compressor#3221: Sequential pipeline offloading](https://github.com/vllm-project/llm-compressor/pull/3221)
- [llm-compressor#3429: Missing unit-offset RMSNorm registry entries](https://github.com/vllm-project/llm-compressor/issues/3429)
