# GLM-5.2 L5: small calibrated oneshot/profile

This example runs the existing QuaRot → FlexSmooth → mixed MXFP4/MXFP8 recipe.
It uses LLMC's SequentialPipeline, CT offload and compressed save. It is a small
calibration run over the **whole model**, not a reduced-layer accuracy experiment.
No GLM weights are downloaded. Run it only on the server with existing weights.

## First server run: single process

Hypothesis: the real checkpoint can pass upstream loading/linearization, the
tested sequential boundaries, floating transforms, MXFP observers and compressed
save/reload within the available memory. L4 does not test that full execution path.

From the feature checkout, use a new output directory on storage with enough room
for the output checkpoint and offload working files:

```bash
set -o pipefail
python examples/glm52_precision/oneshot_profile.py \
  --model /path/to/existing/GLM-5.2-BF16 \
  --output /path/to/output/glm52-l5-smoke \
  --dataset /path/to/local/calibration.json \
  --samples 2 --sequence-length 256 \
  2>&1 | tee glm52-l5-smoke.log
```

Default: NPU, single-process `device_map="auto"`, batch 1, one target per
subgraph, routed expert calibration, BF16 storage and explicit QuaRot FP32
arithmetic. FlexSmooth uses `max_tokens=None`; the small dataset bounds memory.
The MXFP target policy is unchanged. `--dataset` is required: there is no default
external dataset download. Local top-level JSON `list[str]`, JSONL `text` records
and UltraChat `messages` records are accepted. An explicitly supplied HF dataset
name remains supported; `--split` defaults to `train_sft` for that route.

`--batch-size` defaults to 1. `DataCollatorWithPadding` uses the tokenizer's own
pad token and padding side; the driver does not invent or override either. Every
batch is constructed during preflight before model weight loading. A missing pad token fails
there. Larger batches than a rank's dataset are valid.

Before loading weights/CT dispatch, preflight reads local `generation_config.json`
(or derives defaults from local `config.json` if absent), unsets inactive nondefault
`top_p` whenever `do_sample is not True`, and runs strict validation. Tokenizer,
dataset, batches and upstream partition sizes are checked in this same phase.
Local errors are reported to all ranks before weight loading. No checkpoint files
are changed. The loaded model's generation config is sanitized/validated with the
same helper, followed by the existing Flex layout preflight before transforms.
Profile adjustments retain original `top_p`/`do_sample` and distinguish
`pre_load_validation` from `load_validation`; stage timings separate
`pre_load_preflight`, `model_load` and `model_preflight`.
Profiles/logs record `samples_global`,
`samples_local`, `batches_local`, `batch_size`, `tokens_local_valid`,
`tokens_global_valid`, global `sequence_length_min/mean/max`, `world_size`, `rank`,
`device_map`, `flex_max_tokens` and `dataset_source`. Token counts use the attention
mask, excluding padding. Flex's existing activation capture still includes padded
rows; changing batch size can therefore change its search inputs. This task does
not introduce masking or alter the reference math.

ModelSlim reference checkout `1c6391745db71912bb145e7f139db1cff8cad9e1` has
`lab_practice/glm_5_2/glm_5_2_w4a4c8_mxfp4.yaml` select `qwen3_cot_w4a4.json`.
The user reports 42 prompts. Its shared
`msmodelslim/model/common/transformers.py::_get_tokenized_data` tokenizes each
prompt separately without explicit `max_length` or `truncation=True`.
This smoke driver deliberately retains truncation at `--sequence-length` and
`add_special_tokens=False`; format support does not assert identical calibration
tokens to ModelSlim. Tests use invented prompts only; no reference data is bundled
or selected by default. The server operator supplies the dataset.

If placement/working memory fails, restart from the original checkpoint with
`--device-map auto_offload` and a **new** output directory. There is no in-place
resume after a partially applied transform. Optional `--max-memory` accepts an
HF memory map JSON and `--offload-dir` selects the working directory.

Return these files, not the checkpoint:

- `glm52-l5-smoke.log`, including load/save warnings and any traceback;
- `glm52-l5-smoke-profile/rank-0.json` next to the output directory;
- output `config.json` and safetensors index (if sharded).

Pass criteria: process exits successfully; profile status is `passed`; QuaRot and
Flex metadata are `applied`; Flex parameters/losses are finite without an identity
fallback; both MXFP formats occur; reload succeeds and both sampled weight/scale
tiles match exactly. Review the log for missing/unexpected checkpoint parameters.
This does **not** establish full-checkpoint numerical identity, MTP correctness,
vLLM-Ascend compatibility, final accuracy, or acceptable production throughput.

## Measurements

The driver records load, dataset preparation, whole oneshot, QuaRot, calibration
forward, propagation, Flex search/apply, MXFP observer/qparams, save and reload.
`oneshot_total` includes its child stages: do not add it to them. NPU operations
are synchronized at timing boundaries, so this instrumentation adds overhead.

CPU RSS is sampled at stage boundaries; `max_observed_cpu_rss_bytes` is not a
continuous peak. NPU allocated/reserved memory and allocator peaks are recorded
for every visible NPU in a single process, and for the current device per rank in
distributed mode. NPU peaks are cumulative since process start. Process read/write
bytes include other I/O; CT cache counters count **logical** onload/write requests,
not measured PCIe bandwidth or exact physical disk traffic.

Reload releases the transformed model first. In distributed mode every rank saves
through the upstream protocol and exits its process group; only rank 0 reloads.
The reload check reads one 1×32 weight tile and scale from each MXFP format,
without another full-model forward. It does not allocate an extra full-model copy.

## Distributed execution contract

After the single-process report has been reviewed, the same driver accepts
`torchrun`. It sets the local NPU and explicitly initializes `cpu:gloo,npu:hccl`.
Both HCCL socket port ranges default to `auto`; CT's generic `init_dist` is unchanged.
Use at least one sample per rank. Dataset slices use upstream `get_rank_partition`.
Distributed placement is `load_context + auto_offload`, never multi-device `auto`.
Single process also accepts explicit `auto_offload`. These are tested driver
contracts; local CPU tests do not measure accelerator placement or HCCL.

- QuaRot: initialization agrees the plan, seed, precision and shared cache layout.
  All ranks update structural state/metadata. A preparation status reduction waits
  for them, then only the CT source runs norm fusion and all rotations. One final
  status reduction releases calibration. No per-parameter communication. This first
  DDP path requires shared CT CPU/disk transform targets and no prior residents;
  private/device QuaRot backing is explicitly unsupported. Single process keeps
  its existing device placement support.
- Flex: local activation rows remain local; channel MAX and candidate SUM give
  global search decisions. Each mapping first computes out-of-place pending values.
  One status reduction waits for old reads, then the source updates shared backing
  while other ranks copy their local results into current CT residents. Private
  targets use ordinary local updates. One completion status ends the mapping.
  There is no additional result clone or backing reread. Preparation for search
  has one separate status reduction, not one collective per weight.
- A single small CPU status-reduction function is shared. The older per-parameter
  writeback prototype remains independently tested but has **no production callers**.
  The generic global-token-ID prototype is unchanged and not used here. DDP with
  a non-None `max_tokens` is rejected until a minimal global cap is integrated.

Actual lifecycle with explicit `pipeline="sequential"`:

```text
initialize all three modifiers → FX trace
CALIBRATION_START: QuaRot → Flex hooks → Quant observers
each subgraph: floating forward → Flex search/apply → weight observe/qparams
              → propagation with hooks disabled
CALIBRATION_END → finalize → compressed save → reload
```

## Local evidence and remaining server checks

The two-process test runs the real official tiny GLM SequentialPipeline with
shared CPU and disk backing, comparing against a single-process calibration on
the union of both shards. It checks source-only QuaRot, global Flex parameters,
transformed weights, post-Flex observer inputs, resident copies, compressed save
and exact reloaded logits. CPU clones simulate private accelerator residents.
Driver timing and sampled reload helpers also execute in these tests.

The current four-prompt test uses upstream slices `train[0:2]` and `train[2:4]`,
with 7 valid tokens per rank (14 global), batch size 2 and one padded local batch.
Both ranks execute all seven subgraphs in calibration and propagation. A separate
three-sample preflight checks the uneven 1/2 split. Local Flex caps None/128 pass;
distributed None passes, and a finite cap on even one rank fails collectively
before QuaRot. Six real Flex searches each perform a channel MAX and candidate
SUM reductions; alpha/beta, scales and final weights match across ranks exactly.
The union single-rank baseline matches within the existing FP32 tolerance.

Installed compressed-tensors `0.19.1a20260919` was tested through actual
`ModelCompressor.compress_model`, not a stand-in scheduler. Single-process save
compresses all 21 quantized modules sequentially (no module-parallel call).
Two-rank save calls `replace_module_parallel` on both ranks: rank 0 performs 10
real compressions, rank 1 performs 11. The other rank also calls `compress_module`
on meta tensors to prepare the state structure; these are recorded separately.
Each module has exactly one real owner. Both ranks' compressed tensor names,
shapes, dtypes and bytes match. Only rank 0 calls the underlying HF checkpoint
writer. After destroying Gloo, rank 0 reloads independently and matches sampled
MXFP4/MXFP8 tiles, scales and full tiny logits exactly.

Measured real owners below are the same for shared CPU and disk runs. Names are
relative to `model.layers.`; assignment is evidence for this fixture/dependency,
not a hardcoded production schedule.

| Rank 0 (10 modules) | Rank 1 (11 modules) |
| --- | --- |
| `0.mlp.gate_proj` | `0.mlp.up_proj` |
| `0.mlp.down_proj` | `0.self_attn.q_b_proj` |
| `0.self_attn.o_proj` | `0.self_attn.kv_a_proj_with_mqa` |
| `1.self_attn.q_b_proj` | `0.self_attn.indexer.wq_b` |
| `1.self_attn.o_proj` | `1.self_attn.kv_a_proj_with_mqa` |
| `1.self_attn.q_a_proj` | `0.self_attn.q_a_proj` |
| `1.mlp.experts.0.gate_proj` | `1.mlp.experts.0.up_proj` |
| `1.mlp.experts.1.up_proj` | `1.mlp.experts.0.down_proj` |
| `1.mlp.experts.1.down_proj` | `1.mlp.experts.1.gate_proj` |
| `1.mlp.shared_experts.up_proj` | `1.mlp.shared_experts.gate_proj` |
| — | `1.mlp.shared_experts.down_proj` |

Local variable-length tests run the actual three-Modifier SequentialPipeline for
three texts at batch sizes 1, 2 and 4, for both padding sides. Non-padding IDs and
masks are checked, and JSON list/string, JSONL/text and messages input routes
are exercised. The test-only CPU weight-copy hook models private accelerator
residents; the disk run uses unmodified cache behavior. Compression, search,
collectives and save/reload are real in both cases.

**UNVERIFIED:** Ascend MXFP kernels, NPU/HCCL execution and eight-rank scheduling,
memory/I/O and throughput on the real checkpoint. The tiny reload still uses the
GLM-scoped test constructor described below. These CPU results neither establish
real-model precision nor predict the speedup over the reported 3h42 serial save.

2026-09-29 validation: the related QuaRot/Flex/CT integration gate has 345 passing
tests and the same three deferred L4 failures; MXFP plus generation-config save
checks have 34 passing tests. No skips or new failures.
**known pre-existing after structured executor: 3 Flex candidate-loss tolerance
failures** (`test_complete_layer_capture_replay_and_source_composition[0/1]` and
`test_cli_synthetic_cannot_claim_real_l4`). Tolerances and those tests are unchanged.
Machine-readable rank reports include every real/meta compression invocation,
per-mapping alpha/beta/scale, local/global activation maxima and reduction counts.

The disk test exposed nonpersistent rotary buffers omitted from CT's accelerate
disk index. The save wrapper now retains those buffers on the source through the
conversion; it continues using the original parameter compression/saving path.

The local environment lacks torchvision; Transformers 5.17's upstream global MoE
patch scan imports an unrelated Aria image processor and fails before model load.
The driver was additionally exercised on BF16 synthetic weights with the existing
GLM-scoped test constructor, with that override recorded in the report. **The
unmodified production load_context path and NPU execution require the server
smoke; the scoped local run does not prove those paths.** No production loader
workaround, dependency replacement or alternate distributed scheduler is added.

## Change inventory for this integration

Paths below are relative to the repository. Existing algorithm math and the mixed
MXFP recipe are unchanged; ModelSlim source is executed externally as the oracle.

| File | Class/function | Reason |
| --- | --- | --- |
| `src/llmcompressor/modifiers/transform/quarot/base.py` | `QuaRotModifier.on_initialize`, `on_calibration_start` | Agree shared layout once; apply fusion/rotation on source before calibration; keep structural state on all ranks. |
| `src/llmcompressor/modifiers/transform/flex_smooth/base.py` | `FlexSmoothModifier.on_initialize`, `_smooth`, `on_sequential_epoch_end` | Integrate existing global search and uncapped statistics into the real Modifier. |
| `src/llmcompressor/modifiers/transform/flex_smooth/mappings.py` | `apply_scales`, `_prepare_scale_updates` | Prepare a complete mapping before shared writes; update current resident copies without rereading backing. |
| `src/llmcompressor/modifiers/transform/flex_smooth/distributed.py` | Module documentation | Mark the search as integrated; token-ID cap remains standalone. |
| `src/llmcompressor/modifiers/transform/utils/distributed.py` | `finish_transform_phase` | One CPU failed-flag reduction per phase; retain the old per-parameter prototype without production callers. |
| `src/llmcompressor/transformers/compression/compressed_tensors_utils.py` | `modify_save_pretrained` wrapper | Preserve nonpersistent disk-offloaded runtime buffers across accelerate conversion, fixing the failure exposed by the real save path. |
| `tests/flex_smooth/test_distributed_oneshot.py` | `test_distributed_sequential_roundtrip` | Compare two-rank CPU/disk backing against a single-rank union baseline, including lifecycle, observers and reload. |
| `examples/glm52_precision/oneshot_profile.py` | `main`, `calibration_data`, `Profile`, `quantized_tiles` | Deliver the server entrypoint, upstream dataset partition, stage measurements and sampled reload verification. |
| `tools/run_quarot_checks.py` | Gate manifest | Record the driver source hash and distributed integration scope with existing tests. |
| `examples/glm52_precision/README.md` | L5 entry | Point to the runnable driver and current scope. |
| `examples/glm52_precision/distributed_search.md` | Integration status | Supersede the standalone-only search status. |
| `examples/glm52_precision/distributed_offload.md` | Prototype status | Distinguish the retained correctness experiment from the narrower production commit protocol. |
| `examples/glm52_precision/oneshot_partitions.md` | Remaining work | Record integration completion; partition framework remains upstream. |
| `examples/glm52_precision/oneshot_profile.md` | Run contract | Define the hypothesis, experiment, evidence, pass criteria and unverified server behavior. |
