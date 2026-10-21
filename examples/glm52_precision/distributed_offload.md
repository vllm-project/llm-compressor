# Collective transform writeback: local validation

`transform/utils/distributed.py::update_transform_parameter` is a tested writeback
prototype, retained for its independent tests. It is **not used by QuaRot or
FlexSmooth**: their integrated path instead uses source-only data-free QuaRot
and mapping-level Flex commits. See [current oneshot semantics](oneshot_profile.md).
The details below describe the earlier prototype, not the production hot path.

The caller must compute an out-of-place result from the old parameter: modifying
shared storage in place before calling this helper defeats its synchronization.
Every WORLD rank must call the same parameter sequence with compatible replicated
model state and transforms. Onloading must be enabled at entry. This helper does
not establish replica equality, distribute transform computation, or initialize
a process group.

For each existing floating parameter/buffer of unchanged shape and dtype:

1. Snapshot the supplied result, then collectively check local validity/cache kind.
   This also prevents any writer from advancing before the last rank's old read.
2. For CT DistributedCPUCache/DistributedDiskCache, only the CT source rank writes
   shared backing. Private caches update locally. Unmanaged shared CPU storage is
   rejected: `is_shared()` alone cannot prove two ranks refer to the same memory.
3. After write completion, refresh other ranks' resident objects from authoritative
   shared backing. Copy into the existing resident object, preserving references
   held by callers. The writer's resident copy is updated by the existing CT helper.
4. Wait for every refresh before returning. Write/refresh failures produce errors
   on all ranks. There is no rollback; the caller must discard/mark the model failed.

The helper uses the existing CT cache/update APIs. It does not broadcast model
weights or alter cache indexes/checkpoint loading. Without a process group it
delegates directly to the original single-process update helper.

Local validation runs two real CPU Gloo processes with **actual CT shared memory
and disk caches**, plus private CPU parameters. Both possible source ranks,
FP32/BF16, cached/uncached residency, weights and buffers are exercised: 24 cases
per rank, with repeated `2*x+1` updates. Delaying the non-writer's old read checks
that a transform is not accidentally applied twice. Write counters enforce a sole
shared writer; resident references and later backing reloads must equal the
single-application result exactly.

For resident-cache testing, private CPU clones stand in for accelerator copies.
Shared backing is real, but this does **not** test GPU/NPU transfers, streams or
hardware memory peaks. Negative cases cover shape mismatch, unmanaged shared
storage, injected writer failure and injected non-writer refresh failure. Both
ranks must report the same failure. Three more tests cover unchanged single-process
behavior on plain, CPU-offloaded and disk-offloaded modules.

The full checkpoint-free gate passes 222 tests. Evidence includes
`collective-offload.json`, per-rank write counts, fault outcomes, JUnit and source
hashes. Run the standard `tools/run_quarot_checks.py --include-flex-smooth ...` gate.

This first correctness implementation adds a result snapshot and three collectives
per update. A non-writer's existing resident copy is reloaded from backing, which
adds I/O. These costs have not been profiled on the server and must not be marketed
as an eight-device speedup. Those per-parameter costs were intentionally excluded
from the integrated Modifier implementation; generic token-ID integration remains
deferred until after the small uncapped smoke/profile.
