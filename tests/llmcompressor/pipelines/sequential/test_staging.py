import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.offload.module import offload_module, subgraph_onload_modules
from safetensors.torch import save_file

import llmcompressor.pipelines.sequential.staging as staging


@pytest.fixture
def stager():
    if not staging._supports_process_staging():
        pytest.skip("requires Linux process staging")
    with staging.ModuleStager() as instance:
        yield instance


@pytest.fixture
def reader(stager):
    return stager.reader.read_batch


process_only = pytest.mark.skipif(
    not staging._supports_process_staging(), reason="requires Linux process staging"
)


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def make_module(tmp_path, onload="cpu", offload="disk"):
    module = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    module.weight.marker = "metadata"
    module.register_buffer("scalar", torch.tensor(123, dtype=torch.int64))
    module.register_buffer("matrix", torch.arange(6, dtype=torch.float64).view(3, 2).T)
    module.register_buffer("empty", torch.empty(0, 3))
    module.register_buffer("optional", None)
    expected = {key: value.clone() for key, value in module.state_dict().items()}
    kwargs = {"offload_dir": str(tmp_path)} if offload == "disk" else {}
    offload_module(module, onload, offload, **kwargs)
    return module, expected


@pytest.mark.parametrize("pin_memory", [False, True])
def test_disk_module_lifecycle(tmp_path, monkeypatch, pin_memory, stager):
    module, expected = make_module(tmp_path)
    # Offloading itself detaches; staging must preserve the template's flags.
    module._parameters.offloaded_values["weight"].requires_grad_(True)

    def unexpected_stage(*args, **kwargs):
        pytest.fail("Disk staging should use the process reader")

    monkeypatch.setattr(staging.DiskCache, "stage", unexpected_stage)
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(stager.stage, {"layer": module}, pin_memory=pin_memory).result()
    assert module._parameters.is_staged and module._buffers.is_staged
    assert not hasattr(module, "_original_forward_func")
    subgraph_onload_modules({"layer": module})
    assert isinstance(module.weight, torch.nn.Parameter)
    assert module.weight.marker == "metadata"
    assert module.weight.requires_grad
    assert not module.weight.is_pinned()  # CPU execution never pins.
    assert module.optional is None
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


@cuda_only
@pytest.mark.parametrize("pin_memory", [False, True])
def test_async_onload_independent_storage(tmp_path, pin_memory, stager):
    torch.cuda.init()
    stream = torch.cuda.Stream()
    modules = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for _ in range(3):
            module, expected = make_module(tmp_path, onload="cuda:0")
            executor.submit(
                stager.stage, {"layer": module}, pin_memory=pin_memory
            ).result()
            weight = module._parameters.offloaded_values["weight"]
            assert weight.is_pinned() == pin_memory
            with torch.cuda.stream(stream):
                torch.cuda._sleep(10_000_000)
                subgraph_onload_modules({"layer": module})
            modules.append((module, expected))
    stream.synchronize()
    for module, expected in modules:
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[key], rtol=0, atol=0)


def test_shared_shard_and_dtype_conversion(tmp_path, reader):
    path = tmp_path / "shared.safetensors"
    a, b = torch.randn(10, 3), torch.tensor(7, dtype=torch.int64)
    save_file({"a": a, "b": b, "empty": torch.empty(2, 0)}, path)
    results = reader(
        [
            (str(path), name, dtype, False)
            for name, dtype in [
                ("b", torch.int32),
                ("a", torch.bfloat16),
                ("empty", torch.float32),
                ("a", torch.float32),
            ]
        ]
    )
    for actual, expected in zip(results, [b.int(), a.bfloat16(), torch.empty(2, 0), a]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert results[1].data_ptr() != results[3].data_ptr()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    ],
)
def test_supported_dtypes(tmp_path, dtype, reader):
    path = tmp_path / "dtype.safetensors"
    expected = torch.arange(10).to(dtype)
    save_file({"x": expected}, path)
    [actual] = reader([(str(path), "x", dtype, False)])
    assert actual.dtype == dtype
    # Byte comparison covers shell dtypes without arithmetic kernels.
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_failed_read_does_not_publish_partial_module(tmp_path, stager):
    module, _ = make_module(tmp_path)
    buffers = module._buffers.offloaded_values
    path = module._buffers.index[buffers["scalar"]]["safetensors_file"]
    os.unlink(path)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        stager.stage({"layer": module})
    assert not module._parameters.is_staged
    assert not module._buffers.is_staged
    assert module._buffers.offloaded_values is buffers
    assert hasattr(module, "_original_forward_func")


@pytest.mark.parametrize("mode", ["cpu_cache", "custom_cache"])
def test_fallback(tmp_path, stager, mode):
    module, expected = make_module(
        tmp_path, offload="cpu" if mode == "cpu_cache" else "disk"
    )
    if mode == "custom_cache":

        class CustomDiskCache(staging.DiskCache):
            pass

        module._parameters.__class__ = CustomDiskCache
        module._buffers.__class__ = CustomDiskCache
    stager.stage({"layer": module})
    subgraph_onload_modules({"layer": module})
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "header,payload",
    [
        ({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, b"1234"),
        ({"x": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}}, b"1234"),
        ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}, b"12345678"),
        ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 8]}}, b"1234"),
        ({"x": {"dtype": "F32", "shape": [2**62, 4], "data_offsets": [0, 4]}}, b"1234"),
    ],
)
def test_invalid_file(tmp_path, header, payload, reader):
    path = tmp_path / "invalid.safetensors"
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        reader([(str(path), "x", torch.float32, False)])


@pytest.mark.parametrize(
    "encoded",
    [
        b'{"x":{"dtype":"F32","shape":[1],"shape":[1],"data_offsets":[0,4]}}',
        b'{"x":' + b"[" * 17 + b"0" + b"]" * 17 + b"}",
        b"not json",
    ],
)
def test_reject_invalid_headers(tmp_path, encoded, reader):
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"1234")
    with pytest.raises((ValueError, OSError, RuntimeError)):
        reader([(str(path), "x", torch.float32, False)])


@process_only
def test_process_persists_and_mapping_outlives_worker(tmp_path):
    path = tmp_path / "tensor.safetensors"
    expected = torch.arange(128)
    save_file({"x": expected}, path)
    with staging.ModuleStager() as stager:
        worker = stager.reader.process
        assert stager.reader.ready == ("ready", worker.pid, False)
        assert worker.pid != os.getpid()
        request = [(str(path), "x", torch.int64, False)]
        [first] = stager.reader.read_batch(request)
        [second] = stager.reader.read_batch(request)
        assert stager.reader.process.pid == worker.pid
        assert first.data_ptr() != second.data_ptr()
    assert worker.poll() == 0
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)
    first.zero_()
    assert torch.equal(second, expected)


@process_only
def test_worker_death_propagates(tmp_path):
    path = tmp_path / "tensor.safetensors"
    save_file({"x": torch.ones(1)}, path)
    with staging.ModuleStager() as stager:
        stager.reader.process.kill()
        stager.reader.process.wait()
        with pytest.raises(RuntimeError, match="worker exited"):
            stager.reader.read_batch([(str(path), "x", torch.float32, False)])


@process_only
def test_unguarded_script_does_not_execute_in_worker(tmp_path):
    script = tmp_path / "unguarded.py"
    marker = tmp_path / "executions.txt"
    script.write_text(
        "from pathlib import Path\n"
        f"with Path({str(marker)!r}).open('a') as f: f.write('run\\n')\n"
        "from llmcompressor.pipelines.sequential.staging import ModuleStager\n"
        "with ModuleStager() as stager:\n"
        "    assert stager.reader.ready[0] == 'ready'\n"
    )
    subprocess.run([sys.executable, str(script)], check=True, timeout=30)
    assert marker.read_text() == "run\n"


@pytest.mark.parametrize(
    "capability", ["linux", "little_endian", "preadv", "fallocate"]
)
def test_unsupported_platform_uses_upstream(tmp_path, monkeypatch, capability):
    module, expected = make_module(tmp_path)
    if capability in ("linux", "little_endian"):
        monkeypatch.setattr(
            staging,
            "sys",
            SimpleNamespace(
                platform="win32" if capability == "linux" else "linux",
                byteorder="big" if capability == "little_endian" else "little",
            ),
        )
    else:
        attribute = "preadv" if capability == "preadv" else "posix_fallocate"
        monkeypatch.delattr(staging.os, attribute, raising=False)

    def unexpected_worker():
        pytest.fail("Unsupported platforms must not launch a worker")

    monkeypatch.setattr(staging, "ProcessReader", unexpected_worker)
    with staging.ModuleStager() as stager:
        assert stager.reader is None
        stager.stage({"layer": module})
    subgraph_onload_modules({"layer": module})
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


@process_only
def test_worker_recovers_after_read_error(tmp_path, stager):
    path = tmp_path / "tensor.safetensors"
    request = [(str(path), "x", torch.int64, False)]
    with pytest.raises(RuntimeError, match="Staging worker failed"):
        stager.reader.read_batch(request)
    expected = torch.arange(16)
    save_file({"x": expected}, path)
    [actual] = stager.reader.read_batch(request)
    assert torch.equal(actual, expected)


@process_only
def test_shared_memory_directory_and_closed_reader(tmp_path, monkeypatch):
    directory = tmp_path / "shared"
    directory.mkdir()
    monkeypatch.setenv("LLMCOMPRESSOR_STAGING_SHM_DIR", str(directory))
    path = tmp_path / "tensor.safetensors"
    expected = torch.arange(16)
    save_file({"x": expected}, path)
    with staging.ModuleStager() as stager:
        reader = stager.reader
        [actual] = reader.read_batch([(str(path), "x", torch.int64, False)])
    stager.close()  # Cleanup is idempotent.
    assert not list(directory.iterdir())
    assert torch.equal(actual, expected)
    assert reader.process.poll() == 0
    with pytest.raises(RuntimeError, match="closed"):
        reader.read_batch([])
    with pytest.raises(RuntimeError, match="context"):
        stager.stage({})


@process_only
def test_pipeline_worker_cleanup_after_read_error(tmp_path):
    from llmcompressor.pipelines.sequential.pipeline import _submit_subgraph_staging

    module, _ = make_module(tmp_path)
    template = module._parameters.offloaded_values["weight"]
    os.unlink(module._parameters.index[template]["safetensors_file"])
    with pytest.raises(RuntimeError, match="Staging worker failed"):
        with ExitStack() as stack:
            stager = stack.enter_context(staging.ModuleStager())
            worker = stager.reader.process
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=1))
            _, future = _submit_subgraph_staging(
                executor, {"layer": module}, {}, False, stager
            )
            future.result()
    assert future.done()
    assert worker.poll() == 0


@process_only
def test_read_into_handles_short_reads_and_interruptions(tmp_path, monkeypatch):
    from llmcompressor.pipelines.sequential import _staging_io

    path = tmp_path / "bytes"
    path.write_bytes(b"abcdefghij")
    original = os.preadv
    attempts = 0

    def short_read(fd, buffers, offset):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise InterruptedError()
        return original(fd, [buffers[0][:2]], offset)

    monkeypatch.setattr(os, "preadv", short_read)
    result = bytearray(10)
    with path.open("rb") as f:
        _staging_io.read_into(f.fileno(), result, 0)
    assert result == b"abcdefghij"
    assert attempts == 6


@process_only
def test_symlink_shards_are_planned_once(tmp_path, monkeypatch):
    from llmcompressor.pipelines.sequential import _staging_io

    source = tmp_path / "source.safetensors"
    save_file({"a": torch.arange(3), "b": torch.ones(2)}, source)
    a, b = tmp_path / "a", tmp_path / "b"
    a.symlink_to(source)
    b.symlink_to(source)
    original = _staging_io.read_header
    calls = []

    def counted(fd):
        calls.append(fd)
        return original(fd)

    monkeypatch.setattr(_staging_io, "read_header", counted)
    infos, files, size = _staging_io.plan_reads([(str(a), "a"), (str(b), "b")])
    assert len(calls) == len(files) == 1
    assert len(infos) == 2 and size > 0
