import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs

from llmcompressor.observers.min_max import MemorylessMinMaxObserver
from llmcompressor.observers.mse import MemorylessMSEObserver


@pytest.mark.parametrize(
    "observer_cls", [MemorylessMinMaxObserver, MemorylessMSEObserver]
)
@pytest.mark.parametrize("shape", [(1, 1, 128), (1, 4, 128), (2, 1, 256)])
def test_observer_torch_compile(observer_cls, shape):
    args = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="tensor")
    observer = observer_cls(base_name="weight", args=args)
    x = torch.randn(*shape)
    eager_scale, eager_zp = observer(x)
    compiled = torch.compile(observer, fullgraph=True, backend="eager")
    compiled_scale, compiled_zp = compiled(x)
    torch.testing.assert_close(eager_scale, compiled_scale)
    torch.testing.assert_close(eager_zp, compiled_zp)
