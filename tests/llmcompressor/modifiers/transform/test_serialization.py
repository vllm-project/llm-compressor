import pytest

from llmcompressor.modifiers.transform import QuIPModifier, SpinQuantModifier


@pytest.mark.parametrize("modifier", [SpinQuantModifier, QuIPModifier])
def test_reload(modifier):
    instance = modifier(transform_type="hadamard")
    dump = instance.model_dump()
    assert modifier.model_validate(dump) == instance
