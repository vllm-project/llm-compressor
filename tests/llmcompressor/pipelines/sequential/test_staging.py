import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from compressed_tensors.offload.module import offload_module, subgraph_onload_modules
from safetensors.torch import save_file

import llmcompressor.pipelines.sequential.staging as staging

native_only = pytest.mark.skipif(
    staging._staging_native is None,
    reason="optional native staging extension not built",
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


@native_only
@pytest.mark.parametrize("pin_memory", [False, True])
def test_disk_module_lifecycle(tmp_path, monkeypatch, pin_memory):
    module, expected = make_module(tmp_path)
    # Offloading itself detaches; staging must preserve the template's flags.
    module._parameters.offloaded_values["weight"].requires_grad_(True)

    def unexpected_stage(*args, **kwargs):
        pytest.fail("Disk staging should use the native reader")

    monkeypatch.setattr(staging.DiskCache, "stage", unexpected_stage)
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(
            staging.subgraph_stage_modules, {"layer": module}, pin_memory=pin_memory
        ).result()
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


@native_only
@cuda_only
@pytest.mark.parametrize("pin_memory", [False, True])
def test_async_onload_independent_storage(tmp_path, pin_memory):
    torch.cuda.init()
    stream = torch.cuda.Stream()
    modules = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        for _ in range(3):
            module, expected = make_module(tmp_path, onload="cuda:0")
            executor.submit(
                staging.subgraph_stage_modules, {"layer": module}, pin_memory=pin_memory
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


@native_only
def test_shared_shard_and_dtype_conversion(tmp_path):
    path = tmp_path / "shared.safetensors"
    a, b = torch.randn(10, 3), torch.tensor(7, dtype=torch.int64)
    save_file({"a": a, "b": b, "empty": torch.empty(2, 0)}, path)
    results = staging._staging_native.read_batch(
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


@native_only
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
def test_supported_dtypes(tmp_path, dtype):
    path = tmp_path / "dtype.safetensors"
    expected = torch.arange(10).to(dtype)
    save_file({"x": expected}, path)
    [actual] = staging._staging_native.read_batch([(str(path), "x", dtype, False)])
    assert actual.dtype == dtype
    # Byte comparison covers shell dtypes without arithmetic kernels.
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@native_only
def test_failed_read_does_not_publish_partial_module(tmp_path):
    module, _ = make_module(tmp_path)
    buffers = module._buffers.offloaded_values
    path = module._buffers.index[buffers["scalar"]]["safetensors_file"]
    os.unlink(path)
    with pytest.raises(RuntimeError, match="Native staging failed"):
        staging.subgraph_stage_modules({"layer": module})
    assert not module._parameters.is_staged
    assert not module._buffers.is_staged
    assert module._buffers.offloaded_values is buffers
    assert hasattr(module, "_original_forward_func")


@pytest.mark.parametrize("mode", ["disabled", "unavailable", "cpu_cache"])
def test_fallback(tmp_path, monkeypatch, mode):
    if mode == "disabled":
        monkeypatch.setenv("LLMCOMPRESSOR_NATIVE_STAGING", "0")
    elif mode == "unavailable":
        monkeypatch.setattr(staging, "_staging_native", None)
    module, expected = make_module(
        tmp_path, offload="cpu" if mode == "cpu_cache" else "disk"
    )
    staging.subgraph_stage_modules({"layer": module})
    subgraph_onload_modules({"layer": module})
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


@native_only
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
def test_invalid_file(tmp_path, header, payload):
    path = tmp_path / "invalid.safetensors"
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
    with pytest.raises(RuntimeError, match="Native staging failed"):
        staging._staging_native.read_batch([(str(path), "x", torch.float32, False)])


@native_only
@pytest.mark.parametrize(
    "encoded",
    [
        b'{"x":{"dtype":"F32","shape":[1],"shape":[1],"data_offsets":[0,4]}}',
        b'{"x":' + b"[" * 17 + b"0" + b"]" * 17 + b"}",
        b"not json",
    ],
)
def test_reject_invalid_headers(tmp_path, encoded):
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"1234")
    with pytest.raises(RuntimeError, match="Native staging failed"):
        staging._staging_native.read_batch([(str(path), "x", torch.float32, False)])


@native_only
def test_native_call_releases_gil(tmp_path):
    # open() on a FIFO waits for the writer. That writer is another Python
    # thread, so it cannot wake up if read_batch holds the GIL. Run in a child
    # with a timeout so a regression fails without deadlocking the test runner.
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    script = """
import os, sys, threading, time
import torch
from llmcompressor import _staging_native
def writer():
    time.sleep(0.1)
    fd = os.open(sys.argv[1], os.O_WRONLY)
    os.close(fd)
thread = threading.Thread(target=writer)
thread.start()
try:
    _staging_native.read_batch([(sys.argv[1], 'x', torch.float32, False)])
except RuntimeError as exc:
    assert 'regular safetensors file' in str(exc)
else:
    raise AssertionError('FIFO must be rejected')
thread.join()
"""
    subprocess.run([sys.executable, "-c", script, str(fifo)], check=True, timeout=30)
