import functools
import inspect

import pytest
import torch

from llmcompressor.entrypoints.model_free.memory import TensorProfiler
from tests.testing_utils import requires_gpu


def get_n_bytes(*tensors: list[torch.Tensor]):
    return sum(tensor.nbytes for tensor in tensors)


def device_parametrize(test):
    signature = inspect.signature(test)

    @functools.wraps(test)
    def wrapper(*args, _device, **kwargs):
        if _device == "cuda" and not torch.accelerator.is_available():
            pytest.skip("CUDA unavailable")

        with torch.device(_device):
            return test(*args, **kwargs)

    # Tell pytest that this function accepts all of the original test's
    # arguments, plus our hidden `_device` parameter.
    wrapper.__signature__ = signature.replace(
        parameters=[
            *signature.parameters.values(),
            inspect.Parameter(
                "_device",
                inspect.Parameter.KEYWORD_ONLY,
            ),
        ]
    )

    return pytest.mark.parametrize(
        "_device",
        ["meta", "cpu", "cuda"],
    )(wrapper)


@device_parametrize
def test_constructor():
    with TensorProfiler() as prof:
        a = torch.Tensor([0 for _ in range(16)])

    assert prof.memory["total"] == get_n_bytes(a)


@device_parametrize
def test_constructor_functions():
    with TensorProfiler() as prof:
        a = torch.empty(16)
        b = torch.zeros(16)
        c = torch.ones(16)
        d = torch.full((16,), 0)

    assert prof.memory["total"] == get_n_bytes(a, b, c, d)


@device_parametrize
def test_operations():
    with TensorProfiler() as prof:
        a = torch.Tensor([1 for _ in range(16)])
        b = a + a

    assert prof.memory["total"] == get_n_bytes(a, b)


@device_parametrize
def test_views():
    with TensorProfiler() as prof:
        a = torch.empty(16)
        a_storage_bytes = get_n_bytes(a)

        b = a[:8]
        c = a[8:]
        d = a[4:12]  # noqa: F841

        del a
        assert prof.memory["total"] == a_storage_bytes

        del b, c
        assert prof.memory["total"] == a_storage_bytes


@requires_gpu
def test_device_movement():
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda:0")
    meta_device = torch.device("meta")

    with TensorProfiler() as prof:
        a = torch.empty(16, device=cpu_device)
        b = a.to(device=gpu_device)
        c = a.to(device=meta_device)

    assert prof.memory["total"] == get_n_bytes(a, b, c)
    assert prof.memory[cpu_device] == get_n_bytes(a)
    assert prof.memory[gpu_device] == get_n_bytes(b)
    assert prof.memory[meta_device] == get_n_bytes(c)


@device_parametrize
def test_dtype_movement():
    with TensorProfiler() as prof:
        a = torch.empty(16, dtype=torch.float32)
        b = a.to(dtype=torch.bfloat16)
        c = a.to(dtype=torch.float8_e4m3fn)

    assert prof.memory["total"] == get_n_bytes(a, b, c)


@device_parametrize
def test_complex_operations():
    with TensorProfiler() as prof:
        a = torch.randn(32)
        b = torch.randn(32)
        c = a * b
        d = torch.sin(c)
        e = torch.cat([a, b, c, d])

    assert prof.memory["total"] == get_n_bytes(a, b, c, d, e)


@device_parametrize
def test_deletion_tracking():
    with TensorProfiler() as prof:
        a = torch.randn(64)
        b = torch.randn(64)
        c = a + b
        del a
        d = c * 2
        del c
        e = torch.zeros_like(b)
        del b

    assert prof.memory["total"] == get_n_bytes(d, e)


@device_parametrize
def test_view_operations():
    with TensorProfiler() as prof:
        a = torch.randn(4, 4)
        b = a.view(16)  # noqa: F841
        c = a.reshape(2, 8)  # noqa: F841
        d = a.t()  # noqa: F841

    assert prof.memory["total"] == get_n_bytes(a)


@device_parametrize
def test_different_dtypes():
    with TensorProfiler() as prof:
        a = torch.ones(16, dtype=torch.float32)
        b = torch.ones(16, dtype=torch.float64)
        c = torch.ones(16, dtype=torch.int32)
        d = torch.ones(16, dtype=torch.bool)

    assert prof.memory["total"] == get_n_bytes(a, b, c, d)


@device_parametrize
def test_inplace_operations():
    with TensorProfiler() as prof:
        a = torch.randn(16)
        b = torch.randn(16)
        a.add_(b)
        b.mul_(2)

    assert prof.memory["total"] == get_n_bytes(a, b)
