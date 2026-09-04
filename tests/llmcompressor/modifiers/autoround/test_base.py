from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
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


class _FakeAttention(nn.Module):
    """Stand-in for an attention module (e.g. `Qwen3Attention`) that another
    modifier (e.g. `QuantizationModifier`) quantized with an activation-only
    scheme (`weights=None`) in the same `IndependentPipeline` run."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(128, 128)


class _DecoderLayerWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttention()
        self.up_proj = nn.Linear(128, 128)


class _FakeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(128, 128)


class _RegexDecoderLayer(nn.Module):
    """Layer whose target modules are addressed via regex (e.g.
    `re:.*self_attn\\.(q|k|v|o)_proj$`), not by class name, mirroring
    examples/autoround/quantization_wNa16/qwen3_example_mixed_w2a16_w4a16.py."""

    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttention()
        self.mlp = _FakeMlp()
        self.input_layernorm = nn.LayerNorm(128)


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


def test_build_layer_config_for_autoround_supports_mxfp4_activation_groups():
    modifier = AutoRoundModifier(
        ignore=["lm_head"],
        iters=0,
        scheme="MXFP4",
    )
    layer = _FakeDecoderLayer()
    modifier.initialize_quantization(layer)

    wrapped = _wrap_decoding_layer(layer)
    layer_config = modifier._build_layer_config_for_autoround(wrapped)

    assert layer_config == {}


def test_build_layer_config_for_autoround_skips_activation_only_schemes():
    """Regression test: an attention module quantized by a *different* modifier
    (e.g. `QuantizationModifier` targeting `Qwen3Attention` with an
    activation-only scheme, `weights=None`) earlier in the same
    `IndependentPipeline` run must be skipped, not passed to
    `_quant_scheme_to_autoround_config` (which unconditionally accesses
    `weight_args.group_size` and previously raised
    `AttributeError: 'NoneType' object has no attribute 'group_size'`)."""
    modifier = AutoRoundModifier(
        ignore=["lm_head"],
        iters=0,
        targets="Linear",
        scheme="W4A16",
    )
    layer = _DecoderLayerWithAttention()
    modifier.initialize_quantization(layer)

    # Simulate a prior modifier (e.g. QuantizationModifier) having already
    # attached an activation-only quantization_scheme to the attention module.
    layer.self_attn.quantization_scheme = QuantizationScheme(
        targets=["_FakeAttention"],
        weights=None,
        input_activations=QuantizationArgs(num_bits=8, type="float", strategy="tensor"),
    )

    wrapped = _wrap_decoding_layer(layer)
    # Previously raised AttributeError: 'NoneType' object has no attribute
    # 'group_size' when the self_attn module (weights=None) was reached.
    layer_config = modifier._build_layer_config_for_autoround(wrapped)

    assert "model.layers.0.self_attn" not in layer_config


def test_postprocess_qparams_only_applies_autoround_decision_to_autoround_targets():
    """Regression test: `check_to_quantized` inspects AutoRound-only `bits`/
    `act_bits` attributes, so it can only meaningfully judge modules AutoRound
    itself targeted (e.g. `Linear`). Previously it was applied to every module
    in the model, including an attention module quantized by a *different*
    modifier (e.g. `QuantizationModifier` targeting `Qwen3Attention`) earlier
    in the same `IndependentPipeline` run - since that module never gets
    AutoRound's `bits`/`act_bits` attrs, it was always judged "not quantized"
    and had its (unrelated) `quantization_scheme` wiped."""
    modifier = AutoRoundModifier(
        ignore=["lm_head"], iters=0, targets="Linear", scheme="W4A16"
    )
    layer = _DecoderLayerWithAttention()

    layer.up_proj.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, strategy="group", group_size=32),
    )
    layer.self_attn.quantization_scheme = QuantizationScheme(
        targets=["_FakeAttention"],
        weights=None,
        input_activations=QuantizationArgs(num_bits=8, type="float", strategy="tensor"),
    )

    with patch(
        "llmcompressor.modifiers.autoround.base.check_to_quantized",
        return_value=False,
    ):
        modifier._postprocess_qparams(layer, llmc_registered_qparams={})

    # AutoRound decided `up_proj` should NOT stay quantized -> scheme cleared.
    assert not hasattr(layer.up_proj, "quantization_scheme")
    # `self_attn` was never an AutoRound target -> left untouched.
    assert hasattr(layer.self_attn, "quantization_scheme")


def test_get_unquantized_layer_names_matches_regex_targets():
    """Regression test: targets specified as regex patterns (e.g.
    `re:.*self_attn\\.(q|k|v|o)_proj$`) must be resolved via
    `match_named_modules`, not by comparing `module.__class__.__name__`
    (always `"Linear"`) against the target strings, which are never plain
    class names here."""
    modifier = AutoRoundModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=["re:.*self_attn\\.(q|k|v|o)_proj$"],
                weights=QuantizationArgs(num_bits=2, strategy="group", group_size=32),
            ),
            "mlp": QuantizationScheme(
                targets=["re:.*mlp\\.(gate|up|down)_proj$"],
                weights=QuantizationArgs(num_bits=4, strategy="group", group_size=128),
            ),
        },
        ignore=["lm_head"],
        iters=0,
    )
    layer = _RegexDecoderLayer()

    # `mlp.gate_proj` matches the "mlp" target regex and was already quantized.
    layer.mlp.gate_proj.quantization_scheme = QuantizationScheme(
        targets=["re:.*mlp\\.(gate|up|down)_proj$"],
        weights=QuantizationArgs(num_bits=4, strategy="group", group_size=128),
    )
    # `self_attn.q_proj` matches the "attention" target regex but has no
    # quantization_scheme -> should be reported as unquantized.

    unquantized = modifier.get_unquantized_layer_names(layer)

    assert "self_attn.q_proj" in unquantized
    assert "mlp.gate_proj" not in unquantized
    # `input_layernorm` never matches either target regex, even though it
    # also lacks a quantization_scheme -> must not appear.
    assert "input_layernorm" not in unquantized


def test_postprocess_qparams_applies_autoround_decision_with_regex_targets():
    """Regression test: `is_autoround_target` must be computed via
    `match_named_modules` so that regex targets (e.g.
    `re:.*self_attn\\.(q|k|v|o)_proj$`) are recognized as AutoRound targets.
    Previously `module.__class__.__name__ in self.resolved_targets` was always
    `False` for regex targets, so AutoRound's decision to fall a layer back to
    full precision was silently ignored and stale qparams were never cleared."""
    modifier = AutoRoundModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=["re:.*self_attn\\.(q|k|v|o)_proj$"],
                weights=QuantizationArgs(num_bits=2, strategy="group", group_size=32),
            ),
        },
        ignore=["lm_head"],
        iters=0,
    )
    layer = _RegexDecoderLayer()
    layer.self_attn.q_proj.quantization_scheme = QuantizationScheme(
        targets=["re:.*self_attn\\.(q|k|v|o)_proj$"],
        weights=QuantizationArgs(num_bits=2, strategy="group", group_size=32),
    )

    with patch(
        "llmcompressor.modifiers.autoround.base.check_to_quantized",
        return_value=False,
    ):
        modifier._postprocess_qparams(layer, llmc_registered_qparams={})

    # AutoRound decided this regex-matched target should NOT stay quantized.
    assert not hasattr(layer.self_attn.q_proj, "quantization_scheme")


def test_update_device_map_for_dp_uses_current_rank_device():
    modifier = AutoRoundModifier(ignore=["lm_head"], iters=0, scheme="W4A16")
    ar_kwargs = {}

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch(
            "llmcompressor.modifiers.autoround.base.get_local_gpu_group_size",
            return_value=1,
        ),
        patch("torch.accelerator.is_available", return_value=True),
        patch("torch.accelerator.current_device_index", return_value=1),
        patch("torch.accelerator.current_accelerator") as mock_accelerator,
    ):
        mock_accelerator.return_value.type = "cuda"
        modifier._update_device_map_for_dp(ar_kwargs)

    assert ar_kwargs["device_map"] == "cuda:1"


def test_apply_autoround_passes_moved_inputs_to_quantize_block():
    modifier = AutoRoundModifier(ignore=["lm_head"], iters=0, scheme="W4A16")
    layer = _FakeDecoderLayer()
    layer._tmp_name = "decoder"
    modifier._sequential_targets = [layer.__class__.__name__]
    modifier._all_module_input[layer._tmp_name] = [((torch.ones(1),), {})]

    state = MagicMock(spec=State)
    state.model.name_or_path = "stub-model"
    state.model.config = MagicMock()

    autoround = MagicMock()
    autoround.quantize_block.return_value = (None, None)

    with (
        patch.object(
            AutoRoundModifier, "_mapping_config_to_autoround", return_value="W4A16"
        ),
        patch.object(
            AutoRoundModifier, "_build_layer_config_for_autoround", return_value={}
        ),
        patch.object(AutoRoundModifier, "get_unquantized_layer_names", return_value=[]),
        patch.object(AutoRoundModifier, "_preprocess_qparams", return_value={}),
        patch.object(AutoRoundModifier, "_postprocess_qparams"),
        patch.object(
            AutoRoundModifier, "_unwrapper_quantized_layer", side_effect=lambda m: m
        ),
        patch.object(AutoRoundModifier, "_update_device_map_for_dp"),
        patch(
            "llmcompressor.modifiers.autoround.base.align_module_device",
            return_value=nullcontext(),
        ),
        patch(
            "llmcompressor.modifiers.autoround.base.suspend_offloading",
            return_value=nullcontext(),
        ),
        patch(
            "llmcompressor.modifiers.autoround.base.get_local_gpu_group_size",
            return_value=2,
        ),
        patch(
            "llmcompressor.modifiers.autoround.base.get_main_device",
            return_value=torch.device("meta"),
        ),
        patch(
            "llmcompressor.modifiers.autoround.base.AutoRound", return_value=autoround
        ),
    ):
        modifier.apply_autoround(state, [layer])

    quantize_inputs = autoround.quantize_block.call_args.kwargs["inputs"]
    assert quantize_inputs[0][0][0][0].device.type == "cpu"
