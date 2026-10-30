"""Benchmark: pinned vs unpinned weight staging in the sequential pipeline.

Cells (one process each):
  --pin 0              unpinned staging (reads into pageable CPU memory)
  --pin 1              pinned staging (direct reads + async H2D onload)

Run against different compressed-tensors checkouts to compare them, e.g.
origin/main vs a branch; instrumentation hooks that a checkout does not
define are skipped. Workload matches bench_loading.py: Qwen/Qwen3-30B-A3B,
device_map="disk", perfectblend train[:16], max_seq_length=2048, empty recipe.
Weight write-back after each subgraph is disabled by default (--offload none)
to isolate the read path; pass --offload live for production behavior.
"""
import argparse
import json
import os
import shutil
import threading
import time

FOLDER = "/root/.cache/huggingface/offload_folder"

parser = argparse.ArgumentParser()
parser.add_argument("--pin", type=int, choices=[0, 1], required=True)
parser.add_argument("--per-subgraph", type=int, default=1)
parser.add_argument("--offload", choices=["none", "live"], default="none")
parser.add_argument("--out", default="/tmp/opencode/pin")
args = parser.parse_args()

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from llmcompressor import oneshot  # noqa: E402
from llmcompressor.utils import load_context  # noqa: E402

T0 = time.perf_counter()
EVENTS = []
_ev_lock = threading.Lock()
STATS = {
    "reads": {"bytes": 0, "calls": 0, "busy": 0.0},
    "raw_reads": {"bytes": 0, "calls": 0, "busy": 0.0},
    "pin_copy": {"bytes": 0, "calls": 0, "busy": 0.0},
    "pinned_alloc": {"bytes": 0, "calls": 0, "busy": 0.0},
    "onload": {"bytes": 0, "calls": 0, "busy": 0.0},
}
_lock = threading.Lock()


def log_event(kind, dur=0.0, **data):
    with _ev_lock:
        EVENTS.append(
            dict(t=time.perf_counter() - T0, kind=kind, dur=dur,
                 thread=threading.current_thread().name, **data)
        )


def timed(owner, name, kind):
    orig = getattr(owner, name)

    def wrapped(*a, **kw):
        t = time.perf_counter()
        out = orig(*a, **kw)
        log_event(kind, time.perf_counter() - t)
        return out

    setattr(owner, name, wrapped)


def stat_timed(owner, name, key, byte_fn):
    orig = getattr(owner, name, None)
    if orig is None:
        return  # hook not defined by this compressed-tensors checkout

    def wrapped(*a, **kw):
        t = time.perf_counter()
        out = orig(*a, **kw)
        with _lock:
            STATS[key]["bytes"] += byte_fn(out) if out is not None else 0
            STATS[key]["calls"] += 1
            STATS[key]["busy"] += time.perf_counter() - t
        return out

    setattr(owner, name, wrapped)


import compressed_tensors.offload.cache.disk as ct_disk  # noqa: E402
import compressed_tensors.offload.cache.utils as cache_utils  # noqa: E402
import llmcompressor.pipelines.sequential.pipeline as pl  # noqa: E402
from llmcompressor.pipelines.sequential.helpers import Subgraph  # noqa: E402

# phase timing
timed(pl, "subgraph_stage_modules", "stage")
timed(pl, "subgraph_onload_modules", "onload")
timed(pl, "subgraph_offload_modules", "offload")
_orig_fwd = Subgraph.forward


def timed_fwd(self, *a, **kw):
    t = time.perf_counter()
    out = _orig_fwd(self, *a, **kw)
    log_event("forward", time.perf_counter() - t)
    return out


Subgraph.forward = timed_fwd

# per-call stats: disk reads, direct raw reads, the read-then-pin copy,
# pinned allocations, and the H2D onload. Hook names differ across
# compressed-tensors versions; missing hooks are skipped by stat_timed.
stat_timed(ct_disk, "load_disk_tensor_from_offload", "reads",
           lambda out: getattr(out, "nbytes", 0))
for _raw_name in ("_load_raw", "_load_pinned_raw"):
    if hasattr(cache_utils, _raw_name):
        stat_timed(cache_utils, _raw_name, "raw_reads",
                   lambda out: getattr(out, "nbytes", 0) if out is not None else 0)
        break
stat_timed(cache_utils, "_pin_memory", "pin_copy",
           lambda out: getattr(out, "nbytes", 0))
stat_timed(cache_utils, "_empty_pinned", "pinned_alloc",
           lambda out: getattr(out, "nbytes", 0))
_orig_onload = ct_disk.DiskCache.onload


def stat_onload(self, offloaded):
    t = time.perf_counter()
    out = _orig_onload(self, offloaded)
    with _lock:
        if out is not None:
            STATS["onload"]["bytes"] += out.numel() * out.element_size()
            STATS["onload"]["calls"] += 1
            STATS["onload"]["busy"] += time.perf_counter() - t
    return out


ct_disk.DiskCache.onload = stat_onload

if args.offload == "none":
    def _no_offload(modules, offload_kwargs):
        log_event("offload_skipped", 0.0)

    pl.subgraph_offload_modules = _no_offload

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
shutil.rmtree(FOLDER, ignore_errors=True)
os.makedirs(FOLDER, exist_ok=True)
time.sleep(1)

t_load0 = time.perf_counter()
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B", device_map="disk", offload_folder=FOLDER
    )
t_load = time.perf_counter() - t_load0

oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:16]",
    recipe=[],
    max_seq_length=2048,
    num_calibration_samples=16,
    pipeline="sequential",
    sequential_targets_per_subgraph=args.per_subgraph,
    stage_weights_in_pinned_memory=bool(args.pin),
)

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
TOTAL = time.perf_counter() - T0
stages = [e for e in EVENTS if e["kind"] == "stage"]
t_first_stage = min((e["t"] for e in stages), default=0.0)
t_end = max((e["t"] + e["dur"] for e in EVENTS), default=0.0)


def tot(kind):
    return sum(e["dur"] for e in EVENTS if e["kind"] == kind)


report = {
    "pin": bool(args.pin),
    "per_subgraph": args.per_subgraph,
    "offload": args.offload,
    "compressed_tensors": ct_disk.__name__,
    "model_load_s": t_load,
    "total_wall_s": TOTAL,
    "calibration_span_s": t_end - t_first_stage,
    "phases": {
        "stage_s": tot("stage"),
        "onload_s": tot("onload"),
        "forward_s": tot("forward"),
        "offload_s": tot("offload"),
        "n_stage": len(stages),
        "n_forward": sum(1 for e in EVENTS if e["kind"] == "forward"),
    },
    "stats": STATS,
    "events": EVENTS,
}

os.makedirs(args.out, exist_ok=True)
name = f"{'pin' if args.pin else 'nopin'}"
path = os.path.join(args.out, f"{name}_ps{args.per_subgraph}.json")
with open(path, "w") as f:
    json.dump(report, f)

onl = STATS["onload"]
pc = STATS["pin_copy"]
print(f"[{'pin' if args.pin else 'nopin'} ps={args.per_subgraph}] load {t_load:.0f}s "
      f"| span {report['calibration_span_s']:.1f}s | stage "
      f"{report['phases']['stage_s']:.1f}s | onload "
      f"{report['phases']['onload_s']:.1f}s | fwd "
      f"{report['phases']['forward_s']:.1f}s")
print(f"  reads: {STATS['reads']['calls']} calls "
      f"({STATS['raw_reads']['calls']} direct-raw), "
      f"{STATS['reads']['bytes']/1e9:.1f}GB @ "
      f"{STATS['reads']['bytes']/max(STATS['reads']['busy'],1e-9)/1e9:.1f}GB/s")
print(f"  pin copy: {pc['calls']} calls, {pc['bytes']/1e9:.1f}GB in {pc['busy']:.1f}s "
      f"| pinned allocs: {STATS['pinned_alloc']['calls']} "
      f"({STATS['pinned_alloc']['bytes']/1e9:.1f}GB in "
      f"{STATS['pinned_alloc']['busy']:.1f}s)")
print(f"  H2D onload: {onl['calls']} copies, {onl['bytes']/1e9:.1f}GB, "
      f"busy {onl['busy']:.1f}s -> {onl['bytes']/max(onl['busy'],1e-9)/1e9:.1f}GB/s")
print(f"report: {path}")
