import json
import os

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors.torch import load_file

from llmcompressor import model_free_ptq, oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from tests.testing_utils import requires_gpu


def _get_tiny_w4a16_quant():
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            strategy="group",
            group_size=16,
            symmetric=True,
            dynamic=False,
        ),
    )


def _get_tiny_block_quant():
    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type="float",
            strategy="block",
            symmetric=True,
            dynamic=False,
            block_structure=[16, 16],
        ),
    )


@requires_gpu
@pytest.mark.parametrize(
    "scheme", [_get_tiny_w4a16_quant(), "FP8_dynamic", _get_tiny_block_quant()]
)
def test_model_free_ptq_matches_oneshot(scheme, tmp_path):
    model = "nm-testing/tinysmokellama-3.2"
    ignore = ["model.embed_tokens", "lm_head"]
    device = "cuda:0"

    ptq_outdir = tmp_path / "weights_out"
    oneshot_outdir = tmp_path / "oneshot_out"

    model_free_ptq(
        model,
        ptq_outdir,
        scheme=scheme,
        max_workers=2,
        device=device,
        ignore=ignore,
    )

    if isinstance(scheme, str):
        recipe = QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore)
    else:
        config_groups = {"config_group_0": scheme}
        recipe = QuantizationModifier(config_groups=config_groups, ignore=ignore)

    oneshot(
        model=model,
        precision="auto",
        recipe=recipe,
        output_dir=oneshot_outdir,
    )

    ptq_st_files = _get_safetensors_files(ptq_outdir)
    oneshot_st_files = _get_safetensors_files(oneshot_outdir)
    assert set(ptq_st_files) == set(oneshot_st_files)

    for file_name in ptq_st_files:
        _assert_safetensors_equal(ptq_outdir / file_name, oneshot_outdir / file_name)

    _assert_config_equal(ptq_outdir / "config.json", oneshot_outdir / "config.json")


def _get_safetensors_files(dir_path: str) -> list[str]:
    return [
        file_name
        for file_name in os.listdir(dir_path)
        if file_name.endswith("safetensors")
    ]


def _assert_safetensors_equal(a_path: str, b_path: str) -> bool:
    a = load_file(a_path)
    b = load_file(b_path)

    # sometimes models have lm_heads, despite using tied embeddings
    # this can cause oneshot to skip writing the lm_head
    if "lm_head.weight" in a and "lm_head.weight" not in b:
        del a["lm_head.weight"]

    assert a.keys() == b.keys(), (a.keys() - b.keys(), b.keys() - a.keys())

    for key in a.keys():
        value_equal = torch.equal(a[key].to(torch.bfloat16), b[key].to(torch.bfloat16))
        dtype_equal = a[key].dtype == b[key].dtype

        assert value_equal and dtype_equal, (key, value_equal, dtype_equal)


def _assert_config_equal(a_path: str, b_path: str):
    with open(a_path, "r") as f:
        config_a: dict = json.load(f)

    with open(b_path, "r") as f:
        config_b: dict = json.load(f)

    a_qconfig = config_a.pop("quantization_config")
    b_qconfig = config_b.pop("quantization_config")
    config_a.pop("transformers_version")
    config_b.pop("transformers_version")

    assert config_a == config_b

    a_config_groups = a_qconfig.pop("config_groups")
    b_config_groups = b_qconfig.pop("config_groups")
    a_ignore = a_qconfig.pop("ignore")
    b_ignore = b_qconfig.pop("ignore")

    assert set(b_ignore).issubset(set(a_ignore))

    assert len(a_config_groups) == 1
    assert len(b_config_groups) == 1
    a_scheme = list(a_config_groups.values())[0]
    b_scheme = list(b_config_groups.values())[0]

    # TODO: remove this pop after
    # https://github.com/vllm-project/compressed-tensors/pull/489 lands and
    # src/llmcompressor/entrypoints/weights_ptq/helpers.py:34 is removed
    a_scheme["weights"].pop("observer")
    b_scheme["weights"].pop("observer")

    assert a_scheme == b_scheme
