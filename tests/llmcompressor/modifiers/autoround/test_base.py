from unittest.mock import MagicMock, patch

from torch import nn

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.autoround import AutoRoundModifier


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)


def test_on_sequential_epoch_end_passes_all_modules():
    """Verify that on_sequential_epoch_end passes all modules to apply_autoround
    without filtering. Regression test for a bug where an is_module_quantized
    filter silently dropped decoder layers, causing autoround to be a no-op."""
    modifier = AutoRoundModifier(
        ignore=["lm_head"],
        iters=10,
        scheme="W4A16",
    )
    state = MagicMock(spec=State)
    event = Event(type_=EventType.SEQUENTIAL_EPOCH_END)
    modules = [_FakeDecoderLayer(), nn.Linear(64, 64)]

    with (
        patch.object(AutoRoundModifier, "apply_autoround") as mock_apply,
        patch.object(AutoRoundModifier, "post_autoround_cleanup"),
    ):
        modifier.on_sequential_epoch_end(state, event, modules=modules)
        mock_apply.assert_called_once_with(state, modules)
