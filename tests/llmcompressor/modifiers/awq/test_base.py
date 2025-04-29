import pytest
import torch

from llmcompressor.modifiers.awq import AWQMapping, AWQModifier
from llmcompressor.modifiers.factory import ModifierFactory
from tests.llmcompressor.modifiers.conf import setup_modifier_factory


@pytest.mark.unit
class test_awq_is_registered:
    """Ensure AWQModifier is registered in ModifierFactory"""

    setup_modifier_factory()

    modifier = ModifierFactory.create(
        type_="AWQModifier",
        allow_experimental=False,
        allow_registered=True,
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
        ]
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
            "self_attn": self_attn,
            "input_layernorm": torch.nn.LayerNorm(4),
            "mlp": mlp,
        }
    )
    awq._set_resolved_mappings(model)
    for mapping in awq._resolved_mappings:
        if "input_layernorm" in mapping.smooth_name:
            assert set(mapping.balance_names) == {
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            }
            assert set(mapping.balance_layers) == {
                self_attn.q_proj,
                self_attn.k_proj,
                self_attn.v_proj,
            }
            assert mapping.parent_name == "self_attn"
            assert mapping.parent == self_attn
        if "self_attn.v_proj" in mapping.smooth_name:
            assert set(mapping.balance_names) == {"self_attn.o_proj"}
            assert mapping.parent_name == "self_attn.o_proj"
        if "mlp.up_proj" in mapping.smooth_name:
            assert set(mapping.balance_names) == {"mlp.down_proj"}
            assert mapping.parent_name == "mlp.down_proj"

    # make sure we exclude case where o_proj/v_proj shapes are mismatched
    awq = AWQModifier(
        mappings=[
            AWQMapping("re:.*v_proj", ["re:.*o_proj"]),
        ]
    )
    model = torch.nn.ModuleDict(
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
    awq._set_resolved_mappings(model)
    assert len(awq._resolved_mappings) == 0
