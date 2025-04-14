import pytest

from llmcompressor.modifiers.awq import AWQModifier
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
