import pytest
import torch
import torch.fx

from llmcompressor.pipelines.sequential.ast_helpers import autowrap_forward


def _no_wraps_decorator(fn):
    """A decorator that does NOT use functools.wraps.

    This mirrors the shape of ``transformers.integrations.accelerate.
    force_accelerate_hooks`` (see ``transformers/integrations/accelerate.py``),
    which wraps ``forward`` with an inner ``def wrapped(self, *args, **kwargs)``
    without copying ``__wrapped__`` / ``__name__`` onto the wrapper. As a result
    ``inspect.unwrap`` cannot recover the original function and
    ``inspect.getsource`` returns the *wrapper's* source.
    """

    side_effect = {"called": 0}

    def wrapped(self, *args, **kwargs):
        side_effect["called"] += 1
        return fn(self, *args, **kwargs)

    wrapped._side_effect = side_effect
    return wrapped


class _DecoratedForwardModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    @_no_wraps_decorator
    def forward(self, x):
        return self.linear(x)


class _PlainForwardModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.regression
def test_autowrap_forward_handles_unwrapped_decorator():
    """autowrap_forward must not raise KeyError when ``forward`` is wrapped by a
    decorator that omits ``functools.wraps``.

    Regression test for the closure-unwrap bug: previously
    ``inspect.getsource`` returned the wrapper's source (defining a function
    named ``wrapped``), so the ``namespace["forward"]`` lookup raised
    ``KeyError: 'forward'`` and sequential-pipeline calibration could not trace
    such modules at all.
    """
    module = _DecoratedForwardModule()

    with autowrap_forward(module, ignore=[]):
        pass

    # The autowrapped forward should be callable and produce correct output.
    x = torch.randn(2, 4)
    expected = module.linear(x)
    with torch.no_grad():
        out = module(x)
    assert torch.allclose(out, expected)
    # The decorator's side effect must still fire, proving the decorator was
    # re-applied (getsource on the original includes the decorator line, so
    # exec re-applies it).
    assert module.forward._side_effect["called"] == 1


@pytest.mark.regression
def test_autowrap_forward_plain_module_unchanged():
    """autowrap_forward behaviour for an undecorated forward is unchanged."""
    module = _PlainForwardModule()

    with autowrap_forward(module, ignore=[]):
        pass

    x = torch.randn(2, 4)
    expected = module.linear(x)
    with torch.no_grad():
        out = module(x)
    assert torch.allclose(out, expected)
