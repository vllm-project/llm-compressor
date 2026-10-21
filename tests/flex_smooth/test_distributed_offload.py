# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real CT shared backing, with private CPU clones standing in for device caches."""

import json
import os
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from compressed_tensors.distributed.utils import set_source_process
from compressed_tensors.offload import OffloadCache, disable_offloading, offload_module
from compressed_tensors.offload.cache import DistributedCPUCache, DistributedDiskCache

from llmcompressor.modifiers.transform.utils import distributed as writeback


@pytest.mark.parametrize("kind", ["private", "cpu", "disk"])
def test_writeback_single_process_fallback(tmp_path, kind):
    assert not dist.is_initialized()
    model = torch.nn.Linear(4, 4, bias=False)
    model.register_buffer("stat", torch.arange(4).float())
    if kind != "private":
        kwargs = {"offload_dir": str(tmp_path)} if kind == "disk" else {}
        offload_module(model, onload_device="cpu", offload_device=kind, **kwargs)
    with disable_offloading():
        for name in ("weight", "stat"):
            resident = getattr(model, name)
            expected = resident.detach().clone() + 1
            writeback.update_transform_parameter(model, name, expected)
            assert getattr(model, name) is resident
            torch.testing.assert_close(resident, expected, atol=0, rtol=0)


def _module(kind, dtype, directory):
    model = torch.nn.Linear(4, 4, bias=False, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(torch.arange(16).reshape(4, 4))
    model.register_buffer("stat", torch.arange(4).to(dtype))
    if kind != "private":
        kwargs = {"offload_dir": str(directory)} if kind == "disk" else {}
        offload_module(model, onload_device="cpu", offload_device=kind, **kwargs)
        assert isinstance(
            model._parameters,
            DistributedCPUCache if kind == "cpu" else DistributedDiskCache,
        )
    return model


def _resident(model, name, private):
    value = getattr(model, name)
    if private and isinstance(model._parameters, OffloadCache):
        cache = model._parameters if name in model._parameters else model._buffers
        offloaded = cache.offloaded_values[name]
        # CPU onload may alias backing. Force a separate resident object to test
        # the stale-device-copy problem without pretending to run an accelerator.
        value = value.detach().clone()
        cache.keep_onloaded_values[offloaded] = value
    return value


def _worker(rank, store, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=store,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    output, failures = [], {}
    original = writeback.update_offload_parameter
    writes = []

    def record(module, name, data):
        writes.append(name)
        return original(module, name, data)

    writeback.update_offload_parameter = record
    try:
        for owner in (0, 1):
            with set_source_process(owner):
                for kind in ("cpu", "disk", "private"):
                    for dtype in (torch.float32, torch.bfloat16):
                        for resident in (False, True):
                            model = _module(kind, dtype, directory)
                            writes.clear()
                            with disable_offloading() if resident else nullcontext():
                                references = (
                                    {
                                        name: _resident(model, name, True)
                                        for name in ("weight", "stat")
                                    }
                                    if resident
                                    else {}
                                )
                                expected = {
                                    "weight": torch.arange(16).reshape(4, 4).to(dtype),
                                    "stat": torch.arange(4).to(dtype),
                                }
                                for step in range(2):
                                    # Deliberately delay the old read on the non-writer.
                                    if rank != owner:
                                        time.sleep(0.05)
                                    for name in ("weight", "stat"):
                                        updated = getattr(model, name) * 2 + 1
                                        writeback.update_transform_parameter(
                                            model, name, updated
                                        )
                                        expected[name] = expected[name] * 2 + 1
                                        torch.testing.assert_close(
                                            getattr(model, name),
                                            expected[name],
                                            atol=0,
                                            rtol=0,
                                        )
                                        if resident:
                                            assert (
                                                getattr(model, name) is references[name]
                                            )
                                            torch.testing.assert_close(
                                                references[name],
                                                expected[name],
                                                atol=0,
                                                rtol=0,
                                            )
                                assert len(writes) == (
                                    4 if kind == "private" or rank == owner else 0
                                )
                            # Dropping resident values must reveal the same backing.
                            for name in ("weight", "stat"):
                                torch.testing.assert_close(
                                    getattr(model, name), expected[name], atol=0, rtol=0
                                )
                            output.append(
                                {
                                    "owner": owner,
                                    "kind": kind,
                                    "dtype": str(dtype),
                                    "resident": resident,
                                    "writes": len(writes),
                                    "passed": True,
                                }
                            )

        with set_source_process(0):
            for mode in (
                "invalid_shape",
                "write_failure",
                "refresh_failure",
                "unmanaged_shared",
            ):
                model = _module(
                    "private" if mode == "unmanaged_shared" else "disk",
                    torch.float32,
                    directory,
                )
                if mode == "unmanaged_shared":
                    model.weight.share_memory_()
                with disable_offloading():
                    held = _resident(model, "weight", True)
                    cache = model._parameters
                    onload = cache.onload if isinstance(cache, OffloadCache) else None
                    if mode == "write_failure":

                        def fail_write(module, name, data):
                            raise OSError("synthetic write failure")

                        writeback.update_offload_parameter = fail_write
                    if mode == "refresh_failure" and rank == 1:

                        def fail_refresh(value):
                            raise OSError("synthetic refresh failure")

                        cache.onload = fail_refresh
                    data = held + 1
                    if mode == "invalid_shape" and rank == 1:
                        data = data[:1]
                    try:
                        writeback.update_transform_parameter(model, "weight", data)
                    except (ValueError, RuntimeError) as error:
                        failures[mode] = str(error)
                    else:
                        failures[mode] = "DID NOT FAIL"
                    finally:
                        writeback.update_offload_parameter = record
                        if onload is not None:
                            cache.onload = onload
        Path(directory, f"offload-rank-{rank}.json").write_text(
            json.dumps({"cases": output, "failures": failures}, indent=2),
            encoding="utf-8",
        )
    finally:
        writeback.update_offload_parameter = original
        dist.destroy_process_group()


def test_collective_transform_writeback(tmp_path):
    assert dist.is_gloo_available()
    context = mp.spawn(
        _worker,
        args=((tmp_path / "store").as_uri(), str(tmp_path)),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 150
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail(
                    "Offload workers exceeded 150s; possible collective mismatch"
                )
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    ranks = [
        json.loads((tmp_path / f"offload-rank-{rank}.json").read_text())
        for rank in (0, 1)
    ]
    assert ranks[0]["failures"] == ranks[1]["failures"]
    assert all(value != "DID NOT FAIL" for value in ranks[0]["failures"].values())
    assert len(ranks[0]["cases"]) == len(ranks[1]["cases"]) == 24
    if output := os.environ.get("FLEXSMOOTH_REPORT_DIR"):
        Path(output, "collective-offload.json").write_text(
            json.dumps(
                {
                    "real_shared_ct_backing": True,
                    "resident_accelerator_copy_simulated_on_cpu": True,
                    "ranks": ranks,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
