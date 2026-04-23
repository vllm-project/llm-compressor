import torch
from torch import nn

from llmcompressor.modifiers.transform.imatrix.base import (
    IMatrixGatherer,
)

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


class TinyModel(nn.Module):
    """Minimal model with two Linear layers + an lm_head."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.layer1 = nn.Linear(hidden, hidden, bias=False)
        self.layer2 = nn.Linear(hidden, hidden, bias=False)
        self.lm_head = nn.Linear(hidden, 32, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.lm_head(x)
        return x


def _make_gatherer(**kwargs) -> IMatrixGatherer:
    return IMatrixGatherer(**kwargs)


def _run_gatherer(
    model: nn.Module,
    gatherer: IMatrixGatherer,
    inputs: list[torch.Tensor],
):
    """Simulate the lifecycle: initialize → start → forwards → end."""

    # Minimal State-like object
    class _State:
        def __init__(self, m):
            self.model = m
            self.data = None

    state = _State(model)
    gatherer.on_initialize(state)
    gatherer.on_start(state, event=None)

    model.eval()
    with torch.no_grad():
        for x in inputs:
            model(x)

    gatherer.on_end(state, event=None)


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #


class TestBasicCollection:
    """Verify that importance tensors are created with correct
    shape, dtype, and positive values."""

    def test_importance_exists(self):
        model = TinyModel(hidden=16)
        gatherer = _make_gatherer()
        inputs = [torch.randn(2, 4, 16) for _ in range(10)]
        _run_gatherer(model, gatherer, inputs)

        assert hasattr(model.layer1, "_imatrix_importance")
        assert hasattr(model.layer2, "_imatrix_importance")

    def test_shape(self):
        hidden = 16
        model = TinyModel(hidden=hidden)
        gatherer = _make_gatherer()
        inputs = [torch.randn(2, 4, hidden) for _ in range(5)]
        _run_gatherer(model, gatherer, inputs)

        assert model.layer1._imatrix_importance.shape == (hidden,)
        assert model.layer2._imatrix_importance.shape == (hidden,)

    def test_dtype_float32(self):
        model = TinyModel()
        gatherer = _make_gatherer()
        inputs = [torch.randn(1, 4, 16) for _ in range(5)]
        _run_gatherer(model, gatherer, inputs)

        assert model.layer1._imatrix_importance.dtype == torch.float32

    def test_values_positive(self):
        model = TinyModel()
        gatherer = _make_gatherer()
        inputs = [torch.randn(1, 4, 16) for _ in range(10)]
        _run_gatherer(model, gatherer, inputs)

        assert (model.layer1._imatrix_importance > 0).all()
        assert (model.layer2._imatrix_importance > 0).all()


class TestIgnoreList:
    """Verify that ignored layers do NOT get importance attached."""

    def test_lm_head_ignored_by_default(self):
        model = TinyModel()
        gatherer = _make_gatherer()
        inputs = [torch.randn(1, 4, 16) for _ in range(5)]
        _run_gatherer(model, gatherer, inputs)

        assert not hasattr(model.lm_head, "_imatrix_importance")
        assert hasattr(model.layer1, "_imatrix_importance")
        assert hasattr(model.layer2, "_imatrix_importance")

    def test_custom_ignore(self):
        model = TinyModel()
        gatherer = _make_gatherer(ignore=["layer1", "lm_head"])
        inputs = [torch.randn(1, 4, 16) for _ in range(5)]
        _run_gatherer(model, gatherer, inputs)

        assert not hasattr(model.layer1, "_imatrix_importance")
        assert not hasattr(model.lm_head, "_imatrix_importance")
        assert hasattr(model.layer2, "_imatrix_importance")


class TestAccumulationCorrectness:
    """Verify numerical correctness of accumulated statistics."""

    def test_ones_input(self):
        """With input = 1.0 everywhere, E[x²] should be 1.0."""
        model = TinyModel(hidden=8)
        gatherer = _make_gatherer()
        inputs = [torch.ones(1, 4, 8) for _ in range(5)]
        _run_gatherer(model, gatherer, inputs)

        imp = model.layer1._imatrix_importance
        torch.testing.assert_close(
            imp,
            torch.ones(8),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_channel_scaling(self):
        """A channel with 10x input should have 100x importance."""
        hidden = 8
        model = TinyModel(hidden=hidden)
        gatherer = _make_gatherer()

        x = torch.ones(1, 4, hidden)
        x[:, :, 0] = 10.0  # channel 0 is 10x
        inputs = [x.clone() for _ in range(5)]
        _run_gatherer(model, gatherer, inputs)

        imp = model.layer1._imatrix_importance
        ratio = imp[0] / imp[1]
        assert abs(ratio.item() - 100.0) < 1e-3


class TestHooksRemovedAfterEnd:
    """Verify hooks are removed after on_end."""

    def test_hooks_removed(self):
        model = TinyModel()
        gatherer = _make_gatherer()
        inputs = [torch.randn(1, 4, 16) for _ in range(5)]
        _run_gatherer(model, gatherer, inputs)

        # After on_end, observers and hooks should be cleaned up
        assert len(gatherer._observers) == 0
        assert not hasattr(model.layer1, "weight_observer")
        assert not hasattr(model.layer1, "_imatrix_hook")

    def test_no_further_accumulation(self):
        model = TinyModel(hidden=8)
        gatherer = _make_gatherer()
        inputs = [torch.ones(1, 4, 8) for _ in range(5)]
        _run_gatherer(model, gatherer, inputs)

        imp_before = model.layer1._imatrix_importance.clone()

        # Extra forward pass after on_end
        with torch.no_grad():
            model(torch.randn(1, 4, 8) * 1000)

        # Importance should not have changed
        torch.testing.assert_close(model.layer1._imatrix_importance, imp_before)


class TestNoQuantization:
    """Verify that the gatherer does NOT modify model weights."""

    def test_weights_unchanged(self):
        model = TinyModel()
        w1_before = model.layer1.weight.data.clone()
        w2_before = model.layer2.weight.data.clone()
        head_before = model.lm_head.weight.data.clone()

        gatherer = _make_gatherer()
        inputs = [torch.randn(1, 4, 16) for _ in range(10)]
        _run_gatherer(model, gatherer, inputs)

        torch.testing.assert_close(model.layer1.weight.data, w1_before)
        torch.testing.assert_close(model.layer2.weight.data, w2_before)
        torch.testing.assert_close(model.lm_head.weight.data, head_before)


class TestOnEvent:
    """Verify that on_event triggers the lifecycle correctly.

    The real pipeline uses on_event(CALIBRATION_EPOCH_START) to trigger
    on_start, not a direct call. This test ensures the gatherer responds
    to events, which is how oneshot() drives the lifecycle.
    """

    def test_calibration_events_collect_importance(self):
        """on_event must trigger start/end and collect importance."""
        from llmcompressor.core import Event, EventType

        model = TinyModel(hidden=128)
        gatherer = _make_gatherer()

        class _State:
            def __init__(self, m):
                self.model = m

        state = _State(model)
        gatherer.on_initialize(state)

        # Trigger start via event (not direct call)
        start_event = Event(type_=EventType.CALIBRATION_EPOCH_START)
        gatherer.on_event(state, start_event)

        model.eval()
        with torch.no_grad():
            for _ in range(5):
                model(torch.ones(1, 4, 128))

        # Trigger end via event
        end_event = Event(type_=EventType.CALIBRATION_EPOCH_END)
        gatherer.on_event(state, end_event)

        assert hasattr(model.layer1, "_imatrix_importance")
        assert (model.layer1._imatrix_importance > 0).all()

    def test_event_does_not_double_start(self):
        """Sending CALIBRATION_EPOCH_START twice must not crash."""
        from llmcompressor.core import Event, EventType

        model = TinyModel(hidden=128)
        gatherer = _make_gatherer()

        class _State:
            def __init__(self, m):
                self.model = m

        state = _State(model)
        gatherer.on_initialize(state)

        start_event = Event(type_=EventType.CALIBRATION_EPOCH_START)
        gatherer.on_event(state, start_event)
        gatherer.on_event(state, start_event)  # second time — should be no-op

        model.eval()
        with torch.no_grad():
            model(torch.ones(1, 4, 128))

        end_event = Event(type_=EventType.CALIBRATION_EPOCH_END)
        gatherer.on_event(state, end_event)

        assert hasattr(model.layer1, "_imatrix_importance")


class TestRecipeYAML:
    """Verify that the gatherer can be created with recipe-style
    kwargs and that parameters are parsed correctly."""

    def test_default_params(self):
        g = _make_gatherer()
        assert g.ignore == ["lm_head"]
        assert g.targets == ["Linear"]

    def test_custom_params(self):
        g = _make_gatherer(
            ignore=["lm_head", "embed"],
            targets=["Linear"],
        )
        assert "embed" in g.ignore
        assert "lm_head" in g.ignore
