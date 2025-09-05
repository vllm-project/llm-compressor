import shlex
import shutil
import sys
from pathlib import Path
from typing import Callable, NamedTuple

import pytest

from tests.examples.utils import (
    replace_2of4_w4a16_recipe,
    requires_gpu_count,
    verify_2of4_w4a16_output,
    verify_w4a4_fp4_output,
)
from tests.testing_utils import run_cli_command


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
@requires_gpu_count(1)
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

    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}:\n"
            f"Command:\n{shlex.join(command)}\nOutput:\n{result.stdout}"
        )

    if test_case.verify_fn:
        test_case.verify_fn(tmp_path, example_dir)
