import pytest

from llmcompressor.modifiers.transform import QuIPModifier, SpinQuantModifier


@pytest.mark.parametrize("modifier", [SpinQuantModifier, QuIPModifier])
@pytest.mark.parametrize("transform_block_size", [16, 32])
def test_reload(modifier, transform_block_size):
    instance = modifier(
        transform_type="hadamard", transform_block_size=transform_block_size
    )
    dump = instance.model_dump()
    assert modifier.model_validate(dump) == instance
