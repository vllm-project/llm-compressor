from llmcompressor.modifiers.transform import QuIPModifier


def test_reload():
    modifier = QuIPModifier(transform_type="hadamard")
    dump = modifier.model_dump()
    assert QuIPModifier.model_validate(dump) == modifier
