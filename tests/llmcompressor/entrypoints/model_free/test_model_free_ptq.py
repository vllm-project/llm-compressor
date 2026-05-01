import json
import os
import random
from pathlib import Path

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
)
from compressed_tensors.utils.match import match_name
from safetensors.torch import load_file

from llmcompressor import model_free_ptq, oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from tests.testing_utils import requires_gpu


def _get_tiny_w4a16_quant():
    return QuantizationScheme(
        targets=["re:.*self_attn.(q|k|o|v)_proj"],
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
        targets=["re:.*mlp.(down|gate|up)_proj"],
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
    "model,scheme",
    [
        ("Qwen/Qwen3-0.6B", _get_tiny_w4a16_quant()),
        ("Qwen/Qwen3-0.6B", "FP8_dynamic"),
        ("Qwen/Qwen3-0.6B", _get_tiny_block_quant()),
        ("Qwen/Qwen3-0.6B", "NVFP4A16"),
        # Also check an MoE model with 3D tensors
        ("Qwen/Qwen3-30B-A3B", _get_tiny_w4a16_quant()),
    ],
)
def test_model_free_ptq_matches_oneshot(model, scheme, tmp_path):
    ignore = ["model.embed_tokens", "lm_head"]
    device = "cuda:0"

    ptq_outdir = tmp_path / "ptq_out"
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

    _assert_safetensors_index_equal(
        ptq_outdir,
        oneshot_outdir,
    )

    _assert_config_equal(ptq_outdir / "config.json", oneshot_outdir / "config.json")


@requires_gpu
@pytest.mark.parametrize(
    "schemes",
    [(_get_tiny_w4a16_quant(), _get_tiny_block_quant())],
)
def test_stacked_model_free_ptq_matches_oneshot(schemes, tmp_path):
    """
    Test that model_free_ptq can be stacked, also tests that
    model_free_ptq can be run on a pre-existing CT checkpoint
    """

    model = "Qwen/Qwen3-0.6B"
    ignore = ["model.embed_tokens", "lm_head"]
    device = "cuda:0"

    ptq_outdirs = [tmp_path / f"ptq_out_{idx}" for idx in range(len(schemes))]
    oneshot_outdir = tmp_path / "oneshot_out"

    for idx, scheme in enumerate(schemes):
        model_free_ptq(
            model if idx == 0 else ptq_outdirs[idx - 1],
            ptq_outdirs[idx],
            scheme=scheme,
            max_workers=2,
            device=device,
            ignore=ignore,
        )

    config_groups = {
        f"config_group_{idx}": scheme for idx, scheme in enumerate(schemes)
    }
    recipe = QuantizationModifier(config_groups=config_groups, ignore=ignore)

    oneshot(
        model=model,
        precision="auto",
        recipe=recipe,
        output_dir=oneshot_outdir,
    )

    ptq_outdir = ptq_outdirs[-1]
    ptq_st_files = _get_safetensors_files(ptq_outdir)
    oneshot_st_files = _get_safetensors_files(oneshot_outdir)
    assert set(ptq_st_files) == set(oneshot_st_files)

    _assert_safetensors_index_equal(
        ptq_outdir,
        oneshot_outdir,
    )

    _assert_config_equal(ptq_outdir / "config.json", oneshot_outdir / "config.json")


def _get_safetensors_files(dir_path: str) -> list[str]:
    return [
        file_name
        for file_name in os.listdir(dir_path)
        if file_name.endswith("safetensors")
    ]


def _assert_safetensors_index_equal(a_dir: Path, b_dir: Path):
    """
    model_free_ptq and oneshot don't have the same safetensors writing logic,
    such that
    - model_free_ptq could save in 10 model-x-of-00010.safetensors files
    - meanwhile oneshot could save in 15 model-x-of-00015.safetensors files

    Rather than asserting that all safetensors files are the exact same, assert that all
    tensor names are equivalent and verify that a subset of tensor values are the same,
    even if they might live in differently named safetensors files.

    If a single model.safetensors file is found for both, just that will be used
    """

    if os.path.exists(a_dir / "model.safetensors") and os.path.exists(
        b_dir / "model.safetensors"
    ):
        _assert_safetensors_equal(
            a_dir / "model.safetensors", b_dir / "model.safetensors"
        )
        return

    with open(str(a_dir / "model.safetensors.index.json"), "r") as f:
        a_sti_data = json.load(f)
    with open(str(b_dir / "model.safetensors.index.json"), "r") as f:
        b_sti_data = json.load(f)

    # assert all keys are equal (values might be different because they point to
    # different safetensors files)
    weight_map_keys = list(a_sti_data["weight_map"].keys())
    assert set(a_sti_data.keys()) == set(b_sti_data.keys()), "Incompatible keys"
    assert set(weight_map_keys) == set(
        b_sti_data["weight_map"].keys()
    ), "Incompatible weight map keys"

    # assert a subset of randomly selected safetensors are equivalent
    for key in random.sample(weight_map_keys, len(weight_map_keys) // 10):
        a_st_file = a_sti_data["weight_map"][key]
        b_st_file = b_sti_data["weight_map"][key]

        a_tensor = load_file(a_dir / a_st_file)[key]
        b_tensor = load_file(b_dir / b_st_file)[key]

        assert (
            torch.equal(a_tensor, b_tensor) and a_tensor.dtype == b_tensor.dtype
        ), f"key {key} has non-matching tensors {a_tensor} {b_tensor}"


def _assert_safetensors_equal(a_path: str, b_path: str) -> bool:
    a = load_file(a_path)
    b = load_file(b_path)

    # sometimes models have lm_heads, despite using tied embeddings
    # this can cause oneshot to skip writing the lm_head
    if "lm_head.weight" in a and "lm_head.weight" not in b:
        del a["lm_head.weight"]

    assert a.keys() == b.keys(), (
        sorted(a.keys() - b.keys()),
        sorted(b.keys() - a.keys()),
    )

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
    config_a.pop("transformers_version", None)
    config_b.pop("transformers_version", None)
    config_a.pop("torch_dtype", None)
    config_b.pop("torch_dtype", None)
    config_a.pop("dtype", None)
    config_b.pop("dtype", None)
    config_a.pop("layer_types", None)
    config_b.pop("layer_types", None)

    assert config_a == config_b

    a_config_groups = a_qconfig.pop("config_groups")
    b_config_groups = b_qconfig.pop("config_groups")
    a_ignore = a_qconfig.pop("ignore")
    b_ignore = b_qconfig.pop("ignore")

    # QuantizationModifier updates ignore lists with any non-targeted layers
    # model_free_ptq does not. Rather than asserting sets are equal,
    # confirm none conflict with targets
    all_ignores = set(a_ignore).union(set(b_ignore))

    assert len(a_config_groups) == len(b_config_groups)
    a_schemes = list(a_config_groups.values())
    b_schemes = list(b_config_groups.values())

    for a_scheme, b_scheme in zip(a_schemes, b_schemes):
        # TODO: remove this pop after
        # https://github.com/vllm-project/compressed-tensors/pull/489 lands and
        # src/llmcompressor/entrypoints/weights_ptq/helpers.py:34 is removed
        a_scheme["weights"].pop("observer")
        b_scheme["weights"].pop("observer")

        assert a_scheme == b_scheme

        for ignore in all_ignores:
            for target in a_scheme["targets"]:
                assert not match_name(ignore, target)
