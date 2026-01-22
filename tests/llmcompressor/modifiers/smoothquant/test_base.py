import pytest
import torch

from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.smoothquant.base import SmoothQuantModifier


@pytest.mark.unit
@pytest.mark.usefixtures("setup_modifier_factory")
def test_smooth_quant_is_registered():
    smoothing_strength = 0.3
    mappings = [(["layer1", "layer2"], "layer3")]
    modifier = ModifierFactory.create(
        type_="SmoothQuantModifier",
        allow_experimental=False,
        allow_registered=True,
        smoothing_strength=smoothing_strength,
        mappings=mappings,
    )

    assert isinstance(
        modifier, SmoothQuantModifier
    ), "PyTorch SmoothQuant not registered"
    assert modifier.smoothing_strength == smoothing_strength
    assert modifier.mappings == mappings


@pytest.mark.unit
@pytest.mark.usefixtures("setup_modifier_factory")
def test_smooth_quant_defaults():
    default_sq = SmoothQuantModifier()
    assert default_sq.smoothing_strength == 0.5


@pytest.mark.unit
def test_override_defaults():
    strength = 0.7
    dummy_map = [(["layer1", "layer2"], "layer3")]
    non_default_sq = SmoothQuantModifier(
        smoothing_strength=strength, mappings=dummy_map
    )

    assert non_default_sq.smoothing_strength == strength
    assert non_default_sq.mappings == dummy_map


@pytest.mark.unit
def test_moe_all_experts_smoothed():
    """
    Test that SmoothQuant smooths ALL experts in MoE models, not just expert.0.

    Verifies that all experts are included in balance_layers when resolving
    mappings for MoE models with multiple experts.
    """
    num_experts = 8
    hidden_size = 256

    experts = torch.nn.ModuleList(
        [
            torch.nn.ModuleDict(
                {
                    "w1": torch.nn.Linear(hidden_size, hidden_size),
                    "w2": torch.nn.Linear(hidden_size, hidden_size),
                }
            )
            for _ in range(num_experts)
        ]
    )

    model = torch.nn.ModuleDict(
        {
            "layers": torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "input_layernorm": torch.nn.LayerNorm(hidden_size),
                            "mlp": torch.nn.ModuleDict(
                                {
                                    "gate": torch.nn.Linear(hidden_size, num_experts),
                                    "experts": experts,
                                }
                            ),
                        }
                    )
                ]
            )
        }
    )

    sq = SmoothQuantModifier(
        smoothing_strength=0.8,
        mappings=[(["re:.*experts.*w1"], "re:.*input_layernorm")],
        ignore=["re:.*gate"],
    )

    resolved_mappings = sq._resolve_mappings(model)

    assert len(resolved_mappings) == 1
    mapping = resolved_mappings[0]

    assert "input_layernorm" in mapping.smooth_name
    assert (
        len(mapping.balance_layers) == num_experts
    ), f"Expected {num_experts} balance layers, got {len(mapping.balance_layers)}"

    # Verify no duplicates
    balance_layer_ids = [id(layer) for layer in mapping.balance_layers]
    assert len(balance_layer_ids) == len(set(balance_layer_ids))

    # Verify correct layers
    expected_expert_w1s = {experts[i].w1 for i in range(num_experts)}
    assert set(mapping.balance_layers) == expected_expert_w1s


@pytest.mark.unit
def test_moe_multiple_layers_all_experts_smoothed():
    """
    Test SmoothQuant with multiple MoE layers to ensure all experts across
    all layers are smoothed correctly.
    """
    num_layers = 2
    num_experts = 4
    hidden_size = 128

    def create_moe_layer():
        experts = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "w1": torch.nn.Linear(hidden_size, hidden_size),
                        "w2": torch.nn.Linear(hidden_size, hidden_size),
                    }
                )
                for _ in range(num_experts)
            ]
        )

        return torch.nn.ModuleDict(
            {
                "input_layernorm": torch.nn.LayerNorm(hidden_size),
                "mlp": torch.nn.ModuleDict(
                    {
                        "gate": torch.nn.Linear(hidden_size, num_experts),
                        "experts": experts,
                    }
                ),
            }
        )

    model = torch.nn.ModuleDict(
        {"layers": torch.nn.ModuleList([create_moe_layer() for _ in range(num_layers)])}
    )

    sq = SmoothQuantModifier(
        smoothing_strength=0.8,
        mappings=[(["re:.*experts.*w1"], "re:.*input_layernorm")],
        ignore=["re:.*gate"],
    )

    resolved_mappings = sq._resolve_mappings(model)

    assert len(resolved_mappings) == num_layers

    for i, mapping in enumerate(resolved_mappings):
        assert len(mapping.balance_layers) == num_experts, (
            f"Layer {i}: Expected {num_experts} balance layers, "
            f"got {len(mapping.balance_layers)}"
        )

        # Verify all balance layers are unique
        balance_layer_ids = [id(layer) for layer in mapping.balance_layers]
        assert len(balance_layer_ids) == len(set(balance_layer_ids))


@pytest.mark.unit
def test_ignore_behavior():
    """Test that mapping is skipped when ALL layers are in ignore list"""
    hidden_size = 64

    model = torch.nn.ModuleDict(
        {
            "decoder": torch.nn.ModuleDict(
                {
                    "input_layernorm": torch.nn.LayerNorm(hidden_size),
                    "self_attn": torch.nn.ModuleDict(
                        {
                            "q_proj": torch.nn.Linear(hidden_size, hidden_size),
                            "k_proj": torch.nn.Linear(hidden_size, hidden_size),
                            "v_proj": torch.nn.Linear(hidden_size, hidden_size),
                        }
                    ),
                }
            )
        }
    )

    # Test case 1: Some balance layers ignored - mapping should proceed
    sq = SmoothQuantModifier(
        smoothing_strength=0.5,
        mappings=[
            (["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm")
        ],
        ignore=["re:.*q_proj", "re:.*k_proj"],  # Only 2 of 3 balance layers ignored
    )

    resolved_mappings = sq._resolve_mappings(model)
    # Mapping should exist because v_proj is not ignored
    assert len(resolved_mappings) == 1

    # Test case 2: All layers ignored - mapping should be skipped
    sq2 = SmoothQuantModifier(
        smoothing_strength=0.5,
        mappings=[
            (["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm")
        ],
        ignore=[
            "re:.*input_layernorm",
            "re:.*q_proj",
            "re:.*k_proj",
            "re:.*v_proj",
        ],
    )

    resolved_mappings2 = sq2._resolve_mappings(model)
    # Mapping should be skipped because all layers are ignored
    assert len(resolved_mappings2) == 0
