import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from pydantic import ValidationError
from torch.nn import Linear

from llmcompressor.modifiers.awq import AWQMapping, AWQModifier
from llmcompressor.modifiers.awq.base import get_lowest_non_module_list_ancestor
from llmcompressor.modifiers.factory import ModifierFactory


@pytest.mark.unit
@pytest.mark.usefixtures("setup_modifier_factory")
def test_awq_is_registered():
    """Ensure AWQModifier is registered in ModifierFactory"""
    modifier = ModifierFactory.create(
        type_="AWQModifier",
        allow_experimental=False,
        allow_registered=True,
        scheme="W4A16_ASYM",
    )

    assert isinstance(modifier, AWQModifier), "AWQModifier not registered"


@pytest.mark.unit
def test_set_resolved_mappings():
    awq = AWQModifier(
        mappings=[
            AWQMapping(
                "re:.*input_layernorm",
                ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
            ),
            AWQMapping("re:.*v_proj", ["re:.*o_proj"]),
            AWQMapping(
                "re:.*up_proj",
                ["re:.*down_proj"],
            ),
        ],
        scheme="W4A16_ASYM",
    )
    self_attn = torch.nn.ModuleDict(
        {
            "q_proj": Linear(4, 4),
            "k_proj": Linear(4, 4),
            "v_proj": Linear(4, 4),
            "o_proj": Linear(4, 4),
        }
    )
    mlp = torch.nn.ModuleDict(
        {
            "experts": torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "gate_proj": Linear(4, 2),
                            "up_proj": Linear(4, 2),
                            "down_proj": Linear(2, 4),
                        }
                    )
                    for _ in range(3)
                ]
            )
        }
    )
    model = torch.nn.ModuleDict(
        {
            "decoder": torch.nn.ModuleDict(
                {
                    "self_attn": self_attn,
                    "input_layernorm": torch.nn.LayerNorm(4),
                    "mlp": mlp,
                }
            )
        }
    )
    awq._set_resolved_mappings(model)
    for mapping in awq._resolved_mappings:
        if "input_layernorm" in mapping.smooth_name:
            assert set(mapping.balance_names) == {
                "decoder.self_attn.q_proj",
                "decoder.self_attn.k_proj",
                "decoder.self_attn.v_proj",
            }
            assert set(mapping.balance_layers) == {
                self_attn.q_proj,
                self_attn.k_proj,
                self_attn.v_proj,
            }
            assert mapping.parent_name == "decoder.self_attn"
            assert mapping.parent == self_attn
        if "self_attn.v_proj" in mapping.smooth_name:
            assert set(mapping.balance_names) == {"decoder.self_attn.o_proj"}
            assert mapping.parent_name == "decoder.self_attn.o_proj"
        if "mlp.experts" in mapping.smooth_name and "up_proj" in mapping.smooth_name:
            expert_idx = mapping.smooth_name.split(".")[-2]
            expected_down_proj = f"decoder.mlp.experts.{expert_idx}.down_proj"
            assert set(mapping.balance_names) == {expected_down_proj}
            assert mapping.parent_name == expected_down_proj
            assert mapping.parent == mlp["experts"][int(expert_idx)]["down_proj"]

    awq = AWQModifier(
        mappings=[
            # make sure we exclude case where o_proj/v_proj shapes are mismatched
            AWQMapping("re:.*v_proj", ["re:.*o_proj"]),
            # make sure we exclude mapping if any balance layers are skipped
            AWQMapping("re:.*v_proj", ["re:.*z_proj", "re:.*o_proj"]),
        ],
        scheme="W4A16_ASYM",
    )
    model = torch.nn.ModuleDict(
        {
            "decoder": torch.nn.ModuleDict(
                {
                    "self_attn": torch.nn.ModuleDict(
                        {
                            "q_proj": Linear(4, 2),
                            "k_proj": Linear(4, 2),
                            "v_proj": Linear(4, 2),
                            "z_proj": Linear(2, 4),
                            "o_proj": Linear(4, 4),
                        }
                    )
                }
            )
        }
    )
    awq._set_resolved_mappings(model)
    if len(awq._resolved_mappings) > 0:
        assert all(
            "o_proj" not in name for name in awq._resolved_mappings[0].balance_names
        ), "should have skipped v->o mapping because o is incompatible"
        assert all(
            "z_proj" not in name for name in awq._resolved_mappings[0].balance_names
        ), (
            "should have skipped v->[z,o] mapping because o is incompatible even though"
            "z is compatible"
        )
    assert len(awq._resolved_mappings) == 0


@pytest.mark.unit
def test_validate():
    with pytest.raises(ValidationError):
        AWQModifier(scheme="W8A8")

    with pytest.raises(ValidationError):
        AWQModifier(
            config_groups={
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=4,
                        group_size=64,
                    ),
                ),
                "group_1": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=4,
                        group_size=128,
                    ),
                ),
            }
        )

    with pytest.raises(ValidationError):
        AWQModifier(
            config_groups={
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=4,
                        group_size=128,
                    ),
                ),
                "group_1": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=8,
                        group_size=128,
                    ),
                ),
            }
        )

    # valid configuration
    AWQModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(num_bits=4, group_size=128, symmetric=False),
            ),
            "group_1": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(num_bits=4, group_size=128, symmetric=False),
            ),
        }
    )

    AWQModifier(scheme="W4A16", duo_scaling="both")
    with pytest.raises(ValidationError):
        AWQModifier(scheme="W4A16", duo_scaling="Both")
    with pytest.raises(ValidationError):
        AWQModifier(scheme="W4A16", duo_scaling="x")


@pytest.mark.unit
def test_get_lowest_non_module_list_ancestor():
    model = torch.nn.ModuleDict(
        {
            "experts": torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "gate_proj": Linear(4, 2),
                            "down_proj": Linear(2, 4),
                        }
                    )
                    for _ in range(10)
                ]
            )
        }
    )
    
    ancestor_name, ancestor = get_lowest_non_module_list_ancestor(
        "", model
    )
    assert ancestor_name == "" and ancestor == model

    ancestor_name, ancestor = get_lowest_non_module_list_ancestor(
        "experts", model
    )
    assert ancestor_name == "" and ancestor == model

    ancestor_name, ancestor = get_lowest_non_module_list_ancestor(
        "experts.1.gate_proj", model
    )
    assert ancestor_name == "experts.1.gate_proj" and ancestor == model["experts"][1]["gate_proj"]


@pytest.mark.unit
def test_moe_multiple_balance_layers():
    """Test AWQ mapping with multiple balance layers in MoE architecture"""
    awq = AWQModifier(
        mappings=[
            # Map input_layernorm to multiple experts' gate_proj and up_proj
            AWQMapping(
                "re:.*input_layernorm",
                ["re:.*gate_proj", "re:.*up_proj"],
            ),
        ],
        scheme="W4A16_ASYM",
    )

    # Create a simplified MoE model structure
    mlp = torch.nn.ModuleDict(
        {
            "experts": torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "gate_proj": Linear(4, 4),
                            "up_proj": Linear(4, 4),
                            "down_proj": Linear(4, 4),
                        }
                    )
                    for _ in range(2)
                ]
            )
        }
    )
    model = torch.nn.ModuleDict(
        {
            "layer": torch.nn.ModuleDict(
                {
                    "input_layernorm": torch.nn.LayerNorm(4),
                    "mlp": mlp,
                }
            )
        }
    )

    awq._set_resolved_mappings(model)

    # Should have one mapping for input_layernorm
    assert len(awq._resolved_mappings) == 1
    mapping = awq._resolved_mappings[0]

    # Should map to all gate_proj and up_proj across all experts
    expected_balance_names = {
        "layer.mlp.experts.0.gate_proj",
        "layer.mlp.experts.0.up_proj",
        "layer.mlp.experts.1.gate_proj",
        "layer.mlp.experts.1.up_proj",
    }
    assert set(mapping.balance_names) == expected_balance_names

    assert mapping.parent_name == "layer.mlp"
    assert mapping.parent == mlp

