import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from pydantic import ValidationError

from llmcompressor.modifiers.awq import AWQMapping, AWQModifier
from llmcompressor.modifiers.awq.base import get_lowest_common_parent
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
            "q_proj": torch.nn.Linear(4, 4),
            "k_proj": torch.nn.Linear(4, 4),
            "v_proj": torch.nn.Linear(4, 4),
            "o_proj": torch.nn.Linear(4, 4),
        }
    )
    mlp = torch.nn.ModuleDict(
        {
            "up_proj": torch.nn.Linear(4, 10),
            "down_proj": torch.nn.Linear(10, 4),
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
        if "mlp.up_proj" in mapping.smooth_name:
            assert set(mapping.balance_names) == {"decoder.mlp.down_proj"}
            assert mapping.parent_name == "decoder.mlp.down_proj"

    # make sure we exclude case where o_proj/v_proj shapes are mismatched
    awq = AWQModifier(
        mappings=[
            AWQMapping("re:.*v_proj", ["re:.*o_proj"]),
        ],
        scheme="W4A16_ASYM",
    )
    model = torch.nn.ModuleDict(
        {
            "decoder": torch.nn.ModuleDict(
                {
                    "self_attn": torch.nn.ModuleDict(
                        {
                            "q_proj": torch.nn.Linear(4, 2),
                            "k_proj": torch.nn.Linear(4, 2),
                            "v_proj": torch.nn.Linear(4, 2),
                            "o_proj": torch.nn.Linear(4, 4),
                        }
                    )
                }
            )
        }
    )
    awq._set_resolved_mappings(model)
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


@pytest.mark.unit
def test_get_lowest_common_parent():
    mlp = torch.nn.ModuleDict(
        {
            "experts": torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {
                            "gate_proj": torch.nn.Linear(4, 2),
                            "down_proj": torch.nn.Linear(4, 2),
                        }
                    )
                    for _ in range(10)
                ]
            )
        }
    )
    self_attn = torch.nn.ModuleDict(
        {
            "q_proj": torch.nn.Linear(4, 2),
            "k_proj": torch.nn.Linear(4, 2),
            "v_proj": torch.nn.Linear(4, 2),
            "o_proj": torch.nn.Linear(4, 4),
        }
    )
    model = torch.nn.ModuleDict(
        {
            "embed_tokens": torch.nn.Linear(4, 2),
            "decoder": torch.nn.ModuleDict(
                {
                    "self_attn": self_attn,
                    "mlp": mlp,
                }
            ),
        }
    )

    parent_name, parent = get_lowest_common_parent(
        ["decoder.mlp.experts.1.gate_proj", "decoder.mlp.experts.4.down_proj"], model
    )
    assert parent_name == "decoder.mlp" and parent == mlp

    parent_name, parent = get_lowest_common_parent(
        ["decoder.self_attn.q_proj", "decoder.self_attn.v_proj"], model
    )
    assert parent_name == "decoder.self_attn" and parent == self_attn

    parent_name, parent = get_lowest_common_parent(
        ["decoder.mlp.experts.1.gate_proj", "decoder.self_attn.v_proj"], model
    )
    assert parent_name == "decoder" and parent == model["decoder"]

    parent_name, parent = get_lowest_common_parent(
        ["embed_tokens", "decoder.self_attn.v_proj"], model
    )
    assert parent_name == "" and parent == model
