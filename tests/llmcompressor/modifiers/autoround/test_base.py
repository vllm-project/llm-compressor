from unittest.mock import MagicMock, patch

import pytest
from auto_round.schemes import PRESET_SCHEMES as AR_PRESET_SCHEMES
from auto_round.schemes import QuantizationScheme as ARQuantizationScheme
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from torch import nn

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.modifiers.autoround.base import _wrap_decoding_layer


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)


class _MixedFakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 128)
        self.o_proj = nn.Linear(128, 128)
        self.up_proj = nn.Linear(128, 128)


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

    with patch.object(AutoRoundModifier, "apply_autoround") as mock_apply, patch.object(
        AutoRoundModifier, "post_autoround_cleanup"
    ):
        modifier.on_sequential_epoch_end(state, event, modules=modules)
        mock_apply.assert_called_once_with(state, modules)


@pytest.mark.parametrize(
    ("scheme_name", "expected_bits"),
    [
        ("W2A16", 2),
        ("W3A16", 3),
        ("W5A16", 5),
        ("W6A16", 6),
        ("W7A16", 7),
        ("w2a16", 2),
        ("w7a16", 7),
    ],
)
def test_mapping_config_to_autoround_supports_weight_only_wna16_schemes(
    scheme_name, expected_bits
):
    modifier = AutoRoundModifier(
        ignore=["lm_head"],
        iters=0,
        scheme=scheme_name,
    )

    mapped = modifier._mapping_config_to_autoround()

    if scheme_name.upper() in AR_PRESET_SCHEMES:
        assert mapped == scheme_name.upper()
    else:
        assert isinstance(mapped, ARQuantizationScheme)
        assert mapped.bits == expected_bits
        assert mapped.sym is True
        assert mapped.group_size == 128
        assert mapped.data_type == "int"
        assert mapped.act_bits == 16
        assert mapped.act_group_size is None
        assert mapped.act_sym is None
        assert mapped.act_dynamic is None
        assert mapped.act_data_type is None


def test_mapping_config_to_autoround_uses_fallback_for_w7a16():
    assert "W7A16" not in AR_PRESET_SCHEMES

    modifier = AutoRoundModifier(
        ignore=["lm_head"],
        iters=0,
        scheme="W7A16",
    )

    mapped = modifier._mapping_config_to_autoround()

    assert isinstance(mapped, ARQuantizationScheme)
    assert mapped.bits == 7
    assert mapped.group_size == 128


def test_build_layer_config_for_autoround_supports_mixed_weight_only_schemes():
    modifier = AutoRoundModifier(
        ignore=["lm_head"],
        iters=0,
        config_groups={
            "attention": QuantizationScheme(
                targets=["q_proj", "o_proj"],
                weights=QuantizationArgs(num_bits=2, strategy="group", group_size=128),
            ),
            "mlp": QuantizationScheme(
                targets=["up_proj"],
                weights=QuantizationArgs(num_bits=4, strategy="group", group_size=128),
            ),
        },
    )
    layer = _MixedFakeDecoderLayer()
    modifier.initialize_quantization(layer)

    wrapped = _wrap_decoding_layer(layer)
    layer_config = modifier._build_layer_config_for_autoround(wrapped)

    assert "model.layers.0.up_proj" in layer_config
    assert layer_config["model.layers.0.up_proj"]["bits"] == 4
    assert layer_config["model.layers.0.up_proj"]["group_size"] == 128
    assert "model.layers.0.q_proj" not in layer_config
    assert "model.layers.0.o_proj" not in layer_config
