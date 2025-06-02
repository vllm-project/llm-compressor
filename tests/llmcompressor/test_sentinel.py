from llmcompressor.sentinel import Sentinel


def test_sentinel():
    assert Sentinel("MISSING") == Sentinel("MISSING")
    assert Sentinel("MISSING", "module_one") != Sentinel("MISSING", "module_two")
