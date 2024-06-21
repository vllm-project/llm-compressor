import pytest

from llmcompressor.transformers import SparseAutoConfig


@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    [
        "roneneldan/TinyStories-1M",
        "mgoin/TinyStories-1M-ds",
        "zoo:codegen_mono-350m-bigpython_bigquery_thepile-pruned50_quantized",
    ],
)
def test_from_pretrained(tmp_path, pretrained_model_name_or_path):
    assert SparseAutoConfig.from_pretrained(
        pretrained_model_name_or_path, cache_dir=tmp_path
    )
