from __future__ import annotations

import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Callable, List, NamedTuple

import pytest
from transformers import AutoConfig

from tests.testing_utils import requires_gpu, run_cli_command


def replace_2of4_w4a16_recipe(content: str) -> str:
    return content.replace("2of4_w4a16_recipe.yaml", "2of4_w4a16_group-128_recipe.yaml")


def verify_2of4_w4a16_output(tmp_path: Path, example_dir: str):
    output_dir = Path("output_llama7b_2of4_w4a16_channel")

    stages = {
        "quantization": {
            "path": Path("quantization_stage"),
            "format": "marlin-24",
        },
        "sparsity": {
            "path": Path("sparsity_stage"),
            "format": "sparse-24-bitmask",
        },
        "finetuning": {
            "path": Path("finetuning_stage"),
            "format": "sparse-24-bitmask",
        },
    }

    for stage, stage_info in stages.items():
        stage_path = tmp_path / example_dir / output_dir / stage_info["path"]
        recipe_path = stage_path / "recipe.yaml"
        config_path = stage_path / "config.json"

        assert recipe_path.exists(), f"Missing recipe file in {stage}: {recipe_path}"
        assert config_path.exists(), f"Missing config file in {stage}: {config_path}"

        config = AutoConfig.from_pretrained(stage_path)
        assert config is not None, f"Failed to load config in {stage}"

        quant_config = getattr(config, "quantization_config", {})
        if stage == "quantization":
            actual_format = quant_config.get("format")
        else:
            actual_format = quant_config.get("sparsity_config", {}).get("format")

        assert actual_format, f"Missing expected format field in {stage} config"
        assert actual_format == stage_info["format"], (
            f"Unexpected format in {stage}: got '{actual_format}', "
            f"expected '{stage_info['format']}'"
        )


def verify_w4a4_fp4_output(tmp_path: Path, example_dir: str):
    # verify the expected directory was generated
    nvfp4_dirs: List[Path] = [p for p in tmp_path.rglob("*-NVFP4") if p.is_dir()]
    assert (
        len(nvfp4_dirs)
    ) == 1, f"did not find exactly one generated folder: {nvfp4_dirs}"

    # verify the format in the generated config
    config_json = json.loads((nvfp4_dirs[0] / "config.json").read_text())
    config_format = config_json["quantization_config"]["format"]
    assert config_format == "nvfp4-pack-quantized"


class TestCase(NamedTuple):
    path: str
    flags: tuple[str] = ()
    preprocess_fn: None | Callable[[str], str] = None
    # verify_fn(tmp_path, example_dir)
    verify_fn: Callable[[Path, str], None] | None = None

    def __repr__(self):
        values = [f"'{self.path}'"]
        for attr_name in ["flags", "preprocess_fn", "verify_fn"]:
            attr = getattr(self, attr_name)
            if attr:
                if callable(attr):
                    attr_repr = attr.__name__
                else:
                    attr_repr = repr(attr)

                values.append(f"{attr_name}={attr_repr}")

        return f"{self.__class__.__name__}({', '.join(values)})"


@pytest.mark.example
@requires_gpu(1)
@pytest.mark.parametrize(
    "test_case",
    [
        "awq/llama_example.py",
        "awq/qwen3_moe_example.py",
        "big_models_with_sequential_onloading/llama3.3_70b.py",
        "compressed_inference/fp8_compressed_inference.py",
        "quantization_kv_cache/llama3_fp8_kv_example.py",
        "quantization_w4a16/llama3_example.py",
        "quantization_w8a8_fp8/gemma2_example.py",
        "quantization_w8a8_fp8/fp8_block_example.py",
        "quantization_w8a8_fp8/llama3_example.py",
        "quantization_w8a8_int8/llama3_example.py",
        "quantization_w8a8_int8/gemma2_example.py",
        "quantizing_moe/mixtral_example.py",
        pytest.param(
            "quantizing_moe/mixtral_example.py",
            marks=(requires_gpu(2), pytest.mark.multi_gpu),
        ),
        "quantizing_moe/qwen_example.py",
        # sparse_2of4
        "sparse_2of4_quantization_fp8/llama3_8b_2of4.py",
        TestCase(
            "sparse_2of4_quantization_fp8/llama3_8b_2of4.py",
            flags=["--fp8"],
        ),
        TestCase(
            "quantization_2of4_sparse_w4a16/llama7b_sparse_w4a16.py",
            preprocess_fn=replace_2of4_w4a16_recipe,
        ),
        TestCase(
            "quantization_2of4_sparse_w4a16/llama7b_sparse_w4a16.py",
            verify_fn=verify_2of4_w4a16_output,
        ),
        # w4a4_fp4
        TestCase(
            "quantization_w4a4_fp4/llama3_example.py", verify_fn=verify_w4a4_fp4_output
        ),
        TestCase(
            "quantization_w4a4_fp4/llama4_example.py", verify_fn=verify_w4a4_fp4_output
        ),
        TestCase(
            "quantization_w4a4_fp4/qwen_30b_a3b.py", verify_fn=verify_w4a4_fp4_output
        ),
        # skips
        pytest.param(
            "quantizing_moe/deepseek_r1_example.py",
            marks=pytest.mark.skip("exceptionally long run time"),
        ),
        pytest.param(
            "trl_mixin/ex_trl_constant.py",
            marks=pytest.mark.skip("disabled until further updates"),
        ),
        pytest.param(
            "trl_mixin/ex_trl_distillation.py",
            marks=(
                pytest.mark.skip("disabled until further updates"),
                pytest.mark.multi_gpu,
            ),
        ),
    ],
    ids=repr,
)
def test_example_scripts(test_case: str | TestCase, tmp_path: Path):
    if isinstance(test_case, str):
        test_case = TestCase(test_case)

    example_subdir, filename = test_case.path.rsplit("/", 1)
    example_dir = f"examples/{example_subdir}"

    command = [sys.executable, filename]
    if test_case.flags:
        command.extend(test_case.flags)

    script_working_dir = tmp_path / example_dir
    shutil.copytree(Path.cwd() / example_dir, script_working_dir)

    if test_case.preprocess_fn:
        path = script_working_dir / filename
        content = path.read_text(encoding="utf-8")
        content = test_case.preprocess_fn(content)
        path.write_text(content, encoding="utf-8")

    result = run_cli_command(command, cwd=script_working_dir)

    assert result.returncode == 0, (
        f"command failed with exit code {result.returncode}:\n"
        f"Command:\n{shlex.join(command)}\nOutput:\n{result.stdout}"
    )

    if test_case.verify_fn:
        test_case.verify_fn(tmp_path, example_dir)
