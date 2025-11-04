from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from llmcompressor.core import State
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier


@pytest.fixture
def q_config_kwargs(config_0, config_1):
    return dict(
        config_groups=dict(
            group_0=dict(
                targets=["Linear"],
                input_activations=dict(num_bits=8, symmetric=False, strategy="tensor"),
                weights=dict(
                    num_bits=4,
                    symmetric=True,
                    strategy="group",
                    group_size=128,
                    actorder=config_0,
                ),
            ),
            group_1=dict(
                targets=["Linear"],
                input_activations=dict(num_bits=8, symmetric=False, strategy="tensor"),
                weights=dict(
                    num_bits=4,
                    symmetric=True,
                    strategy="group",
                    group_size=128,
                    actorder=config_1,
                ),
            ),
        )
    )


@pytest.fixture
def block_q_config_kwargs():
    return dict(
        config_groups=dict(
            group_block=dict(
                targets=["Linear"],
                input_activations=dict(
                    num_bits=8, symmetric=True, strategy="group", group_size=128
                ),
                weights=dict(
                    num_bits=8,
                    symmetric=True,
                    strategy="block",
                    block_structure=[128, 128],
                ),
            ),
        )
    )


def test_block_strategy_parsing(block_q_config_kwargs):
    modifier = GPTQModifier(**block_q_config_kwargs)
    resolved = modifier.resolve_quantization_config()
    w_scheme = resolved.config_groups["group_block"].weights
    assert w_scheme.strategy == "block"
    assert w_scheme.block_structure == [128, 128]


@pytest.mark.parametrize(
    "has_actorder,actorder,config_0,config_1,expected_0,expected_1",
    [
        # defaults to "static" if nothing provided
        (False, "N/A", None, None, "static", "static"),
        # modifier overrides config if no config provided
        (True, "static", None, None, "static", "static"),
        (True, "group", None, None, "group", "group"),
        (True, None, None, None, None, None),
        # modifier overrides if config partially matches anyways
        (True, "group", None, "group", "group", "group"),
        (True, "group", "group", None, "group", "group"),
        # modifier errors if explicitly conflicts with config
        (True, "static", None, "group", "error", "error"),
        (True, "static", "group", None, "error", "error"),
        (True, "group", None, "static", "error", "error"),
        (True, "group", "static", None, "error", "error"),
        (True, None, "static", None, "error", "error"),
        # modifier overrides to static if nothing is provided
        (False, "N/A", None, "static", "static", "static"),
        (False, "N/A", "static", None, "static", "static"),
        (False, "N/A", "static", "static", "static", "static"),
        # modifier does not override set config vaules
        (False, "N/A", None, "group", "static", "group"),
        (False, "N/A", "group", None, "group", "static"),
        (False, "N/A", "group", "group", "group", "group"),
    ],
)
def test_actorder_resolution(
    has_actorder, actorder, q_config_kwargs, expected_0, expected_1
):
    with pytest.raises(ValueError) if expected_0 == "error" else nullcontext():
        if has_actorder:
            modifier = GPTQModifier(**q_config_kwargs, actorder=actorder)
        else:
            modifier = GPTQModifier(**q_config_kwargs)
        resolved = modifier.resolve_quantization_config()

    if expected_0 != "error":
        assert resolved.config_groups["group_0"].input_activations.actorder is None
        assert resolved.config_groups["group_0"].weights.actorder == expected_0
        assert resolved.config_groups["group_1"].input_activations.actorder is None
        assert resolved.config_groups["group_1"].weights.actorder == expected_1


@pytest.mark.parametrize(
    "strategies,actorder",
    [
        (["group"], None),
        (["group"], "static"),
        (["group"], "group"),
        (["channel", "group"], None),
        (["channel", "group"], "static"),
        (["channel", "group"], "group"),
        (["group", "channel"], None),
        (["group", "channel"], "static"),
        (["group", "channel"], "group"),
    ],
)
def test_config_resolution(strategies, actorder):
    config_groups = {
        str(index): QuantizationScheme(
            targets=[],
            weights=QuantizationArgs(
                strategy=strategy, group_size=(128 if strategy == "group" else None)
            ),
        )
        for index, strategy in enumerate(strategies)
    }

    modifier = GPTQModifier(config_groups=config_groups, actorder=actorder)
    modifier.resolve_quantization_config()

    # validate that actorder was applied
    for config_group in modifier.config_groups.values():
        if config_group.weights.strategy == "group":
            assert config_group.weights.actorder == actorder


@pytest.mark.parametrize(
    "has_actorder,actorder,exp_actorder",
    [
        (False, "N/A", "static"),
        (True, None, None),
        (True, "static", "static"),
        (True, "group", "group"),
    ],
)
def test_serialize_actorder(has_actorder, actorder, exp_actorder):
    if has_actorder:
        modifier = GPTQModifier(targets=["Linear"], scheme="W8A8", actorder=actorder)
    else:
        modifier = GPTQModifier(targets=["Linear"], scheme="W8A8")

    assert modifier.model_dump()["actorder"] == exp_actorder


@pytest.mark.parametrize(
    "scheme,targets,config_groups,resolved_targets,should_error",
    [
        ("W4A16", ["Linear"], None, {"Linear"}, False),
        (
            "W4A16",
            [r"re:.*q_proj$", r"re:.*k_proj$"],
            None,
            {r"re:.*q_proj$", r"re:.*k_proj$"},
            False,
        ),
        (
            None,
            ["Linear"],
            dict(
                group_0=dict(
                    targets=[r"re:.*q_proj$"],
                ),
                group_1=dict(
                    targets=[r"re:.*k_proj$"],
                ),
            ),
            {r"re:.*q_proj$", r"re:.*k_proj$"},
            False,
        ),
        (
            "W4AA16",
            ["Linear"],
            dict(
                group_0=dict(
                    targets=[r"re:.*q_proj$"],
                ),
            ),
            {},
            True,
        ),
    ],
)
def test_resolved_targets(
    scheme, targets, config_groups, should_error, resolved_targets
):
    if should_error:
        with pytest.raises(ValueError):
            GPTQModifier(targets=targets, scheme=scheme, config_groups=config_groups)
    else:
        modifier = GPTQModifier(
            targets=targets, scheme=scheme, config_groups=config_groups
        )

        assert modifier.resolved_targets == resolved_targets


# Group Size Validation Tests


class ModelWithDivisibleLayers(nn.Module):
    """Model with all layers divisible by group_size=128"""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(128, 64)  # 128 columns, divisible
        self.layer2 = nn.Linear(256, 64)  # 256 columns, divisible

    def forward(self, x):
        return self.layer2(self.layer1(x))


class ModelWithNonDivisibleLayers(nn.Module):
    """Model with some layers not divisible by group_size=128"""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(128, 64)  # 128 columns, divisible
        self.layer2 = nn.Linear(100, 64)  # 100 columns, NOT divisible
        self.layer3 = nn.Linear(256, 64)  # 256 columns, divisible
        self.layer4 = nn.Linear(50, 64)   # 50 columns, NOT divisible

    def forward(self, x):
        return self.layer3(self.layer1(x))


class MoEExpertLayer(nn.Module):
    """
    Simple MoE expert layer with 3D weights for testing.
    Mimics fused expert structure: [num_experts, out_features, in_features]
    """

    def __init__(self, num_experts: int, in_features: int, out_features: int):
        super().__init__()
        # 3D weight tensor representing stacked expert weights
        # Shape: [num_experts, out_features, in_features]
        self.weight = nn.Parameter(
            torch.randn(num_experts, out_features, in_features)
        )

    def forward(self, x):
        return x


@pytest.fixture
def group_quant_config():
    """Fixture for group quantization config with group_size=128"""
    return {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 128,
            }
        }
    }


@pytest.fixture
def moe_quant_config():
    """Fixture for MoE quantization config with group_size=128"""
    return {
        "group_0": {
            "targets": ["MoEExpertLayer"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 128,
            }
        }
    }


@pytest.fixture
def channel_quant_config():
    """Fixture for channel quantization config"""
    return {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "channel",
            }
        }
    }


def test_group_size_validation_passes_for_divisible_layers(group_quant_config):
    """Test that validation passes when all layers are divisible by group_size"""
    model = ModelWithDivisibleLayers()
    modifier = GPTQModifier(config_groups=group_quant_config)
    state = State(model=model)

    # Should not raise an error
    modifier.on_initialize(state)


def test_group_size_validation_fails_for_non_divisible_layers(group_quant_config):
    """Test that validation raises error for layers not divisible by group_size"""
    model = ModelWithNonDivisibleLayers()
    modifier = GPTQModifier(config_groups=group_quant_config)
    state = State(model=model)

    # Should raise ValueError with comprehensive error message
    with pytest.raises(ValueError) as exc_info:
        modifier.on_initialize(state)

    error_msg = str(exc_info.value)
    # Verify error message contains key information
    assert "Group size divisibility validation failed" in error_msg
    assert "layer2" in error_msg
    assert "layer4" in error_msg
    assert "100 columns" in error_msg
    assert "50 columns" in error_msg
    assert "ignore:" in error_msg
    assert "vLLM" in error_msg


def test_group_size_validation_can_be_disabled(group_quant_config):
    """Test that validation can be disabled with validate_group_size=False"""
    model = ModelWithNonDivisibleLayers()
    modifier = GPTQModifier(
        config_groups=group_quant_config,
        validate_group_size=False  # Disable validation
    )
    state = State(model=model)

    # Should not raise an error when validation is disabled
    modifier.on_initialize(state)


def test_group_size_validation_not_applied_to_channel_strategy(channel_quant_config):
    """Test that validation is not applied for non-GROUP strategies"""
    model = ModelWithNonDivisibleLayers()
    modifier = GPTQModifier(config_groups=channel_quant_config)
    state = State(model=model)

    # Should not raise an error for CHANNEL strategy
    modifier.on_initialize(state)


def test_group_size_validation_with_ignore_list(group_quant_config):
    """Test that validation passes when problematic layers are in ignore list"""
    model = ModelWithNonDivisibleLayers()
    modifier = GPTQModifier(
        config_groups=group_quant_config,
        ignore=["layer2", "layer4"]  # Ignore problematic layers
    )
    state = State(model=model)

    # Should not raise an error when problematic layers are ignored
    modifier.on_initialize(state)


def test_group_size_validation_suggested_ignore_list_includes_current(
    group_quant_config,
):
    """Test that suggested ignore list includes current ignore list"""
    model = ModelWithNonDivisibleLayers()
    modifier = GPTQModifier(
        config_groups=group_quant_config,
        ignore=["some_other_layer"]  # Existing ignore list
    )
    state = State(model=model)

    with pytest.raises(ValueError) as exc_info:
        modifier.on_initialize(state)

    error_msg = str(exc_info.value)
    # Verify suggested ignore list includes both current and new layers
    assert "some_other_layer" in error_msg
    assert "layer2" in error_msg
    assert "layer4" in error_msg
    # Should show combined list
    assert "['some_other_layer', 'layer2', 'layer4']" in error_msg


def test_group_size_validation_with_quantization_modifier(group_quant_config):
    """Test that validation works with QuantizationModifier (RTN)"""
    model = ModelWithNonDivisibleLayers()
    modifier = QuantizationModifier(config_groups=group_quant_config)
    state = State(model=model)

    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        modifier.on_initialize(state)

    error_msg = str(exc_info.value)
    assert "Group size divisibility validation failed" in error_msg
    assert "layer2" in error_msg
    assert "layer4" in error_msg


def test_group_size_validation_with_partial_ignore(group_quant_config):
    """Test validation when only some problematic layers are ignored"""
    model = ModelWithNonDivisibleLayers()
    modifier = GPTQModifier(
        config_groups=group_quant_config,
        ignore=["layer2"]  # Only ignore one problematic layer
    )
    state = State(model=model)

    # Should still raise error for layer4
    with pytest.raises(ValueError) as exc_info:
        modifier.on_initialize(state)

    error_msg = str(exc_info.value)
    assert "layer4" in error_msg
    # layer2 should not be in the error since it's ignored
    assert "layer2: 100 columns" not in error_msg


@pytest.mark.parametrize(
    "group_size,should_fail",
    [
        (128, True),   # 100 and 50 not divisible by 128
        (64, True),    # 100 and 50 not divisible by 64
        (50, True),    # 128 and 256 not divisible by 50
        (32, True),    # 100 and 50 not divisible by 32
        (2, False),    # All (128, 256, 100, 50) divisible by 2
        (1, False),    # All divisible by 1
    ],
)
def test_group_size_validation_with_different_group_sizes(group_size, should_fail):
    """Test validation with different group_size values"""
    model = ModelWithNonDivisibleLayers()
    config = {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": group_size,
            }
        }
    }
    modifier = GPTQModifier(config_groups=config)
    state = State(model=model)

    if should_fail:
        with pytest.raises(ValueError):
            modifier.on_initialize(state)
    else:
        modifier.on_initialize(state)


def test_group_size_validation_error_message_format(group_quant_config):
    """Test that error message contains all required sections"""
    model = ModelWithNonDivisibleLayers()
    modifier = GPTQModifier(config_groups=group_quant_config)
    state = State(model=model)

    with pytest.raises(ValueError) as exc_info:
        modifier.on_initialize(state)

    error_msg = str(exc_info.value)

    # Check all required sections are present
    assert "ERROR: Group size divisibility validation failed!" in error_msg
    assert "Problematic layers:" in error_msg
    assert "Current ignore list:" in error_msg
    assert "Layers with divisibility issues:" in error_msg
    assert "SUGGESTED FIX:" in error_msg
    assert "ignore:" in error_msg
    assert "validate_group_size: False" in error_msg
    assert "vLLM" in error_msg


# MoE-specific tests


def test_group_size_validation_moe_passes_for_divisible_layers(moe_quant_config):
    """Test that validation passes for MoE layers with divisible input dimensions"""
    # Create model with MoE layers using ModuleDict (similar to AWQ tests)
    model = nn.ModuleDict({
        "mlp": nn.ModuleDict({
            "experts_gate_up": MoEExpertLayer(8, 128, 64),   # 128 divisible by 128
            "experts_down": MoEExpertLayer(8, 256, 64),      # 256 divisible by 128
        })
    })

    modifier = GPTQModifier(config_groups=moe_quant_config)
    state = State(model=model)

    # Should not raise an error
    modifier.on_initialize(state)


def test_group_size_validation_moe_fails_for_non_divisible_layers(moe_quant_config):
    """Test that validation catches MoE layers with non-divisible input dimensions"""
    # Create model with a non-divisible MoE layer
    model = nn.ModuleDict({
        "mlp": nn.ModuleDict({
            "experts_gate_up": MoEExpertLayer(8, 100, 64),   # 100 NOT divisible by 128
            "experts_down": MoEExpertLayer(8, 256, 64),      # 256 divisible by 128
        })
    })

    modifier = GPTQModifier(config_groups=moe_quant_config)
    state = State(model=model)

    # Should raise ValueError for experts_gate_up (100 in_features, not divisible by 128)
    with pytest.raises(ValueError) as exc_info:
        modifier.on_initialize(state)

    error_msg = str(exc_info.value)
    # Verify error message contains information about the problematic MoE layer
    assert "experts_gate_up" in error_msg
    assert "100 columns" in error_msg
    assert "not divisible by group_size=128" in error_msg


def test_group_size_validation_moe_with_ignore_list(moe_quant_config):
    """Test that validation passes when problematic MoE layers are in ignore list"""
    # Create model with a non-divisible MoE layer
    model = nn.ModuleDict({
        "mlp": nn.ModuleDict({
            "experts_gate_up": MoEExpertLayer(8, 100, 64),   # 100 NOT divisible by 128
            "experts_down": MoEExpertLayer(8, 256, 64),      # 256 divisible by 128
        })
    })

    modifier = GPTQModifier(
        config_groups=moe_quant_config,
        ignore=["mlp.experts_gate_up"]  # Ignore the problematic MoE layer
    )
    state = State(model=model)

    # Should not raise an error when problematic layer is ignored
    modifier.on_initialize(state)


def test_group_size_validation_moe_3d_weight_shape():
    """Test that validation correctly identifies columns for 3D MoE weights"""
    # Create a model with MoE layer that has specific dimensions
    # [num_experts=8, out_features=64, in_features=100]
    # in_features=100 is NOT divisible by group_size=128
    model = nn.ModuleDict({
        "experts": MoEExpertLayer(num_experts=8, in_features=100, out_features=64)
    })

    config = {
        "group_0": {
            "targets": ["MoEExpertLayer"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 128,
            }
        }
    }

    modifier = GPTQModifier(config_groups=config)
    state = State(model=model)

    # Should raise ValueError and mention 100 columns (the in_features dimension)
    with pytest.raises(ValueError) as exc_info:
        modifier.on_initialize(state)

    error_msg = str(exc_info.value)
    assert "100 columns" in error_msg


@pytest.mark.parametrize(
    "num_experts,in_features,group_size,should_fail",
    [
        (8, 128, 128, False),  # Divisible
        (8, 256, 128, False),  # Divisible
        (8, 100, 128, True),   # NOT divisible
        (8, 50, 128, True),    # NOT divisible
        (8, 256, 64, False),   # Divisible by 64
        (8, 100, 64, True),    # NOT divisible by 64
        (16, 128, 32, False),  # Divisible by 32
    ],
)
def test_group_size_validation_moe_different_dimensions(
    num_experts, in_features, group_size, should_fail
):
    """Test MoE validation with different expert counts, dimensions, and group sizes"""
    model = nn.ModuleDict({
        "experts": MoEExpertLayer(
            num_experts=num_experts,
            in_features=in_features,
            out_features=64
        )
    })

    config = {
        "group_0": {
            "targets": ["MoEExpertLayer"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": group_size,
            }
        }
    }

    modifier = GPTQModifier(config_groups=config)
    state = State(model=model)

    if should_fail:
        with pytest.raises(ValueError):
            modifier.on_initialize(state)
    else:
        modifier.on_initialize(state)
