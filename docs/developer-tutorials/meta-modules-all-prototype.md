# Whole-model meta prototype

This experiment lives on `meta-modules-all`. It replaces module-attached
restructure metadata with `RestructureHandler`, and delays every module's tensor
state before modifier initialization and tracing.

## Flow

```python
handler = RestructureHandler(model)
handler.prepare_for_restructuring(sequential_targets=targets)
session.initialize(model=model, ...)

# The sequential pipeline receives the same handler.
handler.configure_onload(device)
graph = trace_graph(model, ...)
for subgraph in partition_subgraphs(graph, ...):
    modules = subgraph.submodule_dict(model)
    offload_kwargs = subgraph_onload_modules(modules)
    offload_kwargs.update(handler.onload_and_restructure_modules(modules, device))
    calibrate_and_propagate(subgraph)
    handler.restore_restructured_modules(modules, offload_sources=False)
    subgraph_offload_modules(modules, offload_kwargs)

handler.finish()
```

Non-sequential entrypoints materialize all delayed state before calibration.
Standalone pipeline calls can omit the handler and retain their existing behavior.

## Ownership

- `mapping[current_module]` holds a copied source module with its real tensors or
  original `OffloadCache` objects. Ordinary entries own direct state; restructure
  entries own an entire source subtree.
- Source holders form a separate child graph. Shared modules are copied once;
  child names, optional children, shared parameters, and tensor attribute aliases
  are preserved during preparation.
- Bound methods are rebound to their copied holder. For compressed-tensors
  dispatch, the copied source uses its underlying forward bound to itself;
  normal source onload removes the cache before conversion uses it. Hook
  dictionaries are separate from the live model's dictionaries.
- Existing transforms still run on meta state first and real state at
  materialization. The handler owns source correspondence, transform/context
  records, and temporary norm restoration. No metadata is attached to live modules.
- Materialization preserves initialized module and tensor objects, including
  modifier hooks, registered observer buffers, and newly initialized qparams.
  Observers added after preparation materialize with their owning module.
- Already materialized tied parameters are preserved when another module using
  them is loaded. Consumed source tensor storage is cleared so retained source
  parents do not keep a second copy alive.
- If sequential targets split an experts block, the handler linearizes it upfront
  and delays the converted tensors again. It still initializes and traces on meta,
  and runtime ancestor lookup is unnecessary.

## Review notes and limitations

1. First-pass weight prefetch currently stages live meta modules, rather than
   source holders, so it performs no useful source I/O. Source onload remains
   synchronous. Source-aware staging is a follow-up.
2. Restoring an offset norm installs its retained source holder. This preserves
   the restored architecture and weights, but external references to the original
   pre-preparation norm object are not restored to the model.
3. Known compressed-tensors dispatch and directly bound methods are handled.
   Arbitrary closures or third-party hooks can still capture the original module;
   copying such objects does not rewrite their closures.
4. Tensor attributes in ordinary dictionaries, lists, and tuples are supported.
   Custom containers and value-dependent initialization paths need further work.
   Newly added meta state without a source uses the existing zero/scale-one
   defaults, which cannot represent every possible modifier-specific initializer.
5. Copying tensor data into initialized targets retains the existing temporary
   allocation cost. Sources shared by delayed aliases can retain their original
   storage until those aliases are consumed. This prototype does not attempt a
   general zero-copy transfer.
6. Individual expert targets still hit the routing control-flow tracing error
   shared with main; upfront conversion does not fix that independent problem.

CPU checks cover shared modules and parameters, tensor attributes, meta observer
buffers, norm restoration, and staged/unstaged CPU and disk source caches. Eleven
tiny seeded comparisons match main exactly in logits, model state, and packed
exports. These checks do not establish GPU/DDP behavior or large-model memory use.
