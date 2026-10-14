# RFC: Modifier Initialization for Lazy MoE Linearization

- **Status:** Draft for discussion
- **Date:** 2026-10-13

## Summary

Lazy MoE linearization replaces a fused experts module with newly constructed
expert `Linear` modules while sequential calibration is already running. Those
modules did not exist when modifiers initialized the model or started
calibration. Any modifier that attaches configuration, observers, hooks, or
module-identity bookkeeping at those lifecycle points can therefore miss the
new experts.

There are two main approaches: initialize newly created modules through an
`on_modules_added` callback, or defer module-specific setup and initialize each
subgraph as the sequential pipeline reaches it. This RFC compares those
approaches and examines the layerwise setup already present on
`origin/kylesayrs/sep10`.

## Context and current behavior

The sequential pipeline currently has this order:

1. The session initializes modifiers against the model as loaded.
2. `LifecycleCallbacks.calibration_start()` runs once.
3. For each subgraph, the pipeline onloads it, lazily linearizes any MoE in
   that subgraph, calls `LifecycleCallbacks.modules_added(...)`, and then runs
   the first calibration forward.

The timing is visible in `src/llmcompressor/pipelines/sequential/pipeline.py`.
Linearization constructs new expert modules in
`src/llmcompressor/modeling/moe/linear_experts.py`; it does not transfer
modifier hooks from the replaced fused module.

`LifecycleCallbacks.modules_added` already exists in
`src/llmcompressor/core/session_functions.py`. It dispatches to modifiers that
implement `on_modules_added`. `QuantizationModifier` has such a handler and
uses `QuantizationMixin.initialize_new_modules` to apply resolved schemes and,
when calibration is active, observers and ordinary quantization calibration
hooks. The handler is intentionally safe when given modules that already have
quantization state.

GPTQ is a concrete gap. `GPTQModifier` uses `QuantizationMixin` directly and
does not implement `on_modules_added`. It builds `_module_names` in
`on_initialize` and registers its Hessian hook in `on_calibration_start`.
When GPTQ supplies the quantization configuration, its newly linearized experts
also miss the quantization state that would have been attached during model
initialization. If another modifier attaches that generic state, GPTQ still
misses the Hessian hook and its name map remains stale; compression uses that
map to identify modules. The generic quantization initialization helper does
not install the GPTQ-specific hook.

The issue is broader than GPTQ if lazy linearization is intended to work with
arbitrary recipes. For example, SparseGPT/Wanda register hooks by traversing
target layers at calibration start; REAP attaches hooks to the experts module
and its existing expert children; AutoRound and mapping-based transforms also
capture module references or initialize quantization state before the lazy
replacement. Each needs a compatibility decision based on its own semantics.

## Goals

- Ensure every targeted module created by lazy linearization has the correct
  modifier-owned state before its first calibration forward.
- Avoid resetting state or registering duplicate hooks on existing modules.
- Preserve sequential onloading/offloading behavior and the guarantee that
  calibration runs before any module is compressed or repacked.
- Avoid silent partial processing: unsupported modifier combinations should
  use eager linearization or fail clearly.
- Keep model transformation code independent of modifier-specific policy.

## Terminology

"Layerwise initialization" here means initializing the modules in the current
subgraph immediately before that subgraph's calibration forward. It does not
mean rerunning `session.initialize()` or the complete modifier lifecycle for
every layer. Recipe parsing, target resolution, and other model-wide setup can
still happen once; module-specific configuration, observers, and hooks are
applied when the relevant subgraph is onloaded.

## `kylesayrs/sep10` as precedent

The inspected ref is `origin/kylesayrs/sep10` at `af26b93fb`; its layerwise
decompression/compression path was introduced in `9e984ba6a`. It does include
per-subgraph quantization setup, but it is not lazy MoE linearization:
`entrypoints/oneshot.py` calls `linearize_moe(self.model)` before
`session.initialize()` when it finds an unlinearized MoE.

That branch's `layerwise_decompression` mode is for recalibrating a
pre-compressed model. The sequential pipeline still calls
`LifecycleCallbacks.calibration_start()` once. It then decompresses the current
subgraph's already-quantized MLP modules, calls modifier-specific
`start_layerwise_calibration(model, modules)`, calibrates them, and removes
their hooks/freezes their state at the subgraph end. `QuantizationMixin` applies
the resolved config to `allowed_modules` and initializes observers/hooks;
GPTQ adds its Hessian hook in its own `start_layerwise_calibration` method.

So the branch is a concrete implementation of scoped layerwise setup for
quantization and GPTQ, not a general layerwise session initialization. It
demonstrates that Option 2 need not be a full lifecycle rewrite. It is not a
drop-in solution for lazy MoE, though: the branch eagerly linearizes MoEs, and
its current target selection begins with modules that are already quantized.
Fresh expert linears need to be selected by their resolved targets and names;
GPTQ must also register their names because its existing `_module_names` map was
built before those new module identities existed.

## Options

### Option 1: Initialize modules when they are added

Keep the existing whole-model `on_initialize` and one-time
`on_calibration_start` behavior. After lazy linearization creates expert
modules, call `on_modules_added` so each modifier can apply its normal
module-specific setup to the new modules only.

**Major changes:** GPTQ needs an `on_modules_added` handler that uses
`initialize_new_modules` for quantization state, updates `_module_names`, and
installs its Hessian hooks. The current `QuantizationModifier` handler is
already a precedent for the generic quantization portion. This does not mean
every modifier needs a handler: only modifiers whose behavior depends on the
new modules need late-module support. Other affected modifiers need their own
handler or an explicit incompatibility policy.

The callback currently receives the full subgraph module map rather than a
strict list of additions. Either change the pipeline to send a delta, or make
handlers filter the map and tolerate repeated notifications. GPTQ must
separately track which identities have Hessian hooks; an existing quantization
scheme alone does not prove that GPTQ setup ran.

**Pros:** Smallest change to the existing lifecycle; normal initialization
continues to handle all modules already present. The callback runs after
linearization and before the new modules' first forward. It reuses the current
dispatcher and quantization helper, and does not require moving ordinary
initialization for existing modules.

**Cons:** Each affected modifier needs correct late-module behavior, which can
duplicate some setup logic and requires careful filtering/idempotency with the
current subgraph-wide payload. This is a targeted catch-up mechanism, not a
general layerwise initialization model; it does not reduce setup for modules
that already exist.

### Option 2: Initialize each subgraph as the pipeline reaches it

Resolve recipe configuration and perform model-wide setup once, but defer
module-specific configuration, observers, and calibration hooks until the
subgraph is onloaded. After lazy linearization, initialize all eligible
modules in that subgraph—both original modules and newly created expert
linears—before its first forward. Complete or clean up per-subgraph state at
the corresponding sequential-epoch end.

This should be a scoped per-subgraph setup API (for example,
`start_layerwise_calibration`), not a repeated call to
`session.initialize()` or an unmodified replay of every modifier's
`on_calibration_start` callback. The pipeline must pass fully qualified names
as well as module objects so targets and modifier maps can be resolved.

**Major changes:** Split modifier work into model-wide one-time setup and
per-module/per-subgraph setup. Quantization needs to apply the resolved config
to the current allowed modules, initialize observers and hooks, and clean
those hooks up after the subgraph. GPTQ additionally needs Hessian hooks and
to add the new module names to its compression map. Other modifiers need an
equivalent scoped setup if they participate in the layerwise contract. This
does not remove modifier-specific work; it changes that work from handling
late additions to handling the active module scope.

The selected pipeline must drive the scope setup. If sequential calibration
defers module-local setup while other pipelines retain full-model setup, that
choice must be represented somewhere. It need not be implemented as scattered
entrypoint checks: the selected pipeline can invoke the scoped lifecycle API,
while other pipelines invoke it once for the full model (or retain an
equivalent full-model path). However, because `session.initialize()` currently
runs before pipeline selection and performs modifier initialization, that
initialization must be split or coordinated with the pipeline choice.

The `kylesayrs/sep10` branch already has a scoped quantization setup helper
and a GPTQ wrapper for its layerwise decompression mode. Adapting that pattern
is likely moderate work for ordinary quantization plus GPTQ, but the helper's
current selection assumes existing quantized modules. Lazy-created linears
must instead be selected using resolved targets, and GPTQ's names must be
registered when those modules are introduced. The remaining larger cost is
supporting other modifiers and separating their global from layerwise state.

**Pros:** Naturally includes newly linearized children in the active subgraph;
the pipeline need not identify which modules are new. Treating old and new
modules uniformly can provide a clearer sequential setup boundary. Per-module
state can be limited to the active subgraph, aligning with onloading/offloading
and potentially reducing live hooks or observers.

**Cons:** More invasive than initializing only late modules. Participating
modifiers must support the split setup/cleanup protocol, and model-wide setup
must be separated from module-local setup. This is not a way to avoid
modifier-specific logic. Modifiers with cross-layer statistics, global mapping
resolution, or hooks on parent modules need special treatment. Overlapping or
persistent modules also require clear rules for whether their state is retained
or recreated. Pipeline-specific behavior must be coordinated with
initialization, though that coordination can live behind pipeline or lifecycle
APIs rather than requiring ad hoc entrypoint conditionals.

## Comparison and recommendation

| | Option 1: modules added | Option 2: layerwise setup |
| --- | --- | --- |
| Modules receiving setup | Newly created modules only. | All eligible modules in the active subgraph, old and new. |
| Modifier changes | Late-module handler only for affected modifiers. | Scoped setup/cleanup for participating modifiers; separate global from module-local work. |
| Added-module delta | Useful, but not required; today's full-subgraph payload requires filtering/idempotency. | Not needed; requires a named subgraph scope instead. |
| Lifecycle impact | Keeps the existing initialize and calibration-start flow. | Changes when module-local initialization runs and coordinates it with the selected pipeline. |
| Entrypoint impact | Usually contained in the transformation/pipeline callback path. | Pipeline-aware dispatch is needed, but can be encapsulated by pipelines/lifecycle APIs rather than scattered checks. |
| Initial implementation scope | Lower for a small supported set of modifiers. | Moderate for quantization/GPTQ; broader for general modifier support. |
| Best fit | A targeted fix for lazy-created children. | A deliberate sequential, layerwise initialization model. |

For the immediate goal of lazy MoE linearization, Option 1 is the least
disruptive if only a small set of modifiers must be supported: it preserves
the current lifecycle and asks only affected modifiers to catch up new
modules. Option 2 offers a more uniform subgraph boundary and avoids
new-versus-existing module detection, but requires a broader split between
one-time and module-scoped modifier work. It is not automatically a larger
entrypoint rewrite: pipeline-aware coordination is necessary, but can be
encapsulated in the pipeline/lifecycle contract. `sep10` demonstrates that
one-time session setup plus scoped per-subgraph modifier setup is workable; it
does not eliminate the need to split setup or adapt participating modifiers.
The key scope decision is whether the goal is targeted support for lazy-created
children, or a general layerwise contract for sequential calibration.

Eager linearization remains a fallback for unsupported modifier combinations,
not a third lazy-initialization design. Copying observer state or hook handles
from a fused module is not a general substitute: fused modules commonly lack
matching child observers, and hooks are attached to specific module objects.
Modifier setup should resolve configuration for the new modules' qualified
names. Initialization should remain in the pipeline/modifier layer rather
than coupling `linearize_moe` to the active recipe.

## Compatibility audit

| Modifier/family | Current late-module risk | Decision needed |
| --- | --- | --- |
| `QuantizationModifier` | Has `on_modules_added` and uses `initialize_new_modules`. | Retain and test the existing callback; layerwise setup is another option. |
| GPTQ | No module-added handler; name map and Hessian hooks are created before lazy linears exist. | Add either scoped callback setup or a layerwise setup method; both must maintain names and Hessian hooks. |
| SparseGPT/Wanda | Target layers and calibration hooks are set up before new expert linears are inserted. | Add callback/layerwise statistic setup or require eager linearization. |
| REAP | Hooks are attached to the current experts module/children, which lazy conversion can replace. | Rebind during module addition or layerwise setup; verify MoE-specific semantics. |
| AutoRound | Quantization state is initialized before lazy linears; its capture/calibration path is custom. | Audit separately; do not inherit generic observer setup blindly. |
| AWQ, SmoothQuant, and other mapping-based transforms | Mappings may capture module objects or resolve against the pre-linearized tree. | Verify each transform; support, adapt mappings, or use eager conversion. |

This table is an initial risk inventory, not a claim that every listed
modifier is currently supported by the lazy sequential path.

## Open questions

- Which approach should be implemented first: a late-module callback for
  GPTQ/quantization, or a reusable per-subgraph setup path?
- Is the first supported scope GPTQ plus ordinary quantization, or should lazy
  linearization support all calibration/pruning modifiers that target expert
  linears?
- Should `modules_added` receive only the newly created/replaced module delta,
  or should modifier handlers remain responsible for filtering an idempotent
  subgraph-wide map?
- Should unsupported modifier combinations automatically fall back to eager
  linearization, or fail with a clear compatibility error unless the user
  explicitly opts in?
- Should modifier compatibility be declared through a capability field, or
  maintained as an explicit allowlist until the modifier ecosystem is audited?

## Validation criteria

- A GPTQ lazy-MoE run initializes every targeted expert projection before its
  first sequential forward, accumulates Hessians once per forward, compresses
  all eligible experts, and does not fail on stale module names during
  compression.
- Under Option 1, repeated/no-op `modules_added` notifications do not duplicate
  hooks, reset observers, or alter modules that were initialized earlier.
- Under Option 2, per-subgraph hooks are active before that subgraph's first
  forward and cleaned up at its end without resetting global modifier state.
- Repacking does not silently leave targeted experts dense. Unsupported recipe
  combinations fail clearly or take the eager path.
- Generic quantization behavior, fake-quantization disabling during the
  sequential forward, and onload/offload bookkeeping remain intact.
- Modifier-specific cleanup removes hooks and releases per-module state at
  calibration end.
