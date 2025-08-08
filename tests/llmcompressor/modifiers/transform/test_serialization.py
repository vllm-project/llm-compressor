import pytest

from llmcompressor.modifiers.transform import QuIPModifier


@pytest.mark.parametrize("modifier", [QuIPModifier])
def test_reload(modifier):
    instance = modifier(transform_type="hadamard")
    dump = instance.model_dump()
    assert modifier.model_validate(dump) == instance
