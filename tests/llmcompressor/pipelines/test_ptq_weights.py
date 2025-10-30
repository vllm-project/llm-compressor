import os

import pytest
import torch
from safetensors.torch import load_file

from llmcompressor import oneshot, ptq_weights
from llmcompressor.modifiers.quantization import QuantizationModifier
from tests.testing_utils import requires_gpu


@requires_gpu
@pytest.mark.parametrize("scheme", ["FP8_dynamic", "NVFP4A16"])
def test_weights_ptq_e2e(scheme, tmp_path):
    model = "nm-testing/tinysmokellama-3.2"
    ptq_ignore = ["model.embed_tokens.weight", "lm_head.weight", "re:.*norm.weight$"]
    oneshot_ignore = ["lm_head"]
    device = "cuda:0"

    ptq_outdir = tmp_path / "weights_out"
    oneshot_outdir = tmp_path / "oneshot_out"

    ptq_weights(
        model,
        ptq_outdir,
        scheme=scheme,
        max_workers=2,
        device=device,
        ignore=ptq_ignore,
    )

    oneshot(
        model=model,
        recipe=QuantizationModifier(
            targets="Linear", scheme=scheme, ignore=oneshot_ignore
        ),
        output_dir=oneshot_outdir,
    )

    ptq_st_files = _get_safetensors_files(ptq_outdir)
    oneshot_st_files = _get_safetensors_files(oneshot_outdir)
    assert set(ptq_st_files) == set(oneshot_st_files)

    for file_name in ptq_st_files:
        _assert_safetensors_equal(ptq_outdir / file_name, oneshot_outdir / file_name)


def _get_safetensors_files(dir_path: str) -> list[str]:
    return [
        file_name
        for file_name in os.listdir(dir_path)
        if file_name.endswith("safetensors")
    ]


def _assert_safetensors_equal(a_path: str, b_path: str) -> bool:
    a = load_file(a_path)
    b = load_file(b_path)

    assert a.keys() == b.keys(), (a.keys() - b.keys(), b.keys() - a.keys())

    for key in a.keys():
        value_equal = torch.equal(a[key].to(torch.bfloat16), b[key].to(torch.bfloat16))
        dtype_equal = a[key].dtype == b[key].dtype

        assert value_equal and dtype_equal, (key, value_equal, dtype_equal)
